from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
from textwrap import dedent
from loguru import logger

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from caller import AutoCaller
from api_models import RETRY_CONFIG
from utils import parse_json_response


CLUSTER_PROMPT = dedent("""
    You will be given a long list of descriptions of different textual attributes. These textual attributes are ones that commonly appear in language model responses to user prompts under the following broad theme:

    <user_prompt_theme_summary>
    {cluster_summary}
    </user_prompt_theme_summary>

    Your task is to cluster these attributes into clusters. To do this, go through the list of attributes from top to bottom, while maintaining a running list of clusters.

    - If the new attribute is semantically similar to a previous cluster (i.e. responses containing the new attribute will likely contain the attributes in the cluster), then add the new attribute to that cluster. Only do this when the attributes are truly highly similar. If the new attribute and the old cluster share some similarities but could be realized differently in a response, do not add it to the cluster and instead create a new cluster.

    - If the new attribute is not semantically similar to any previous cluster, create a new cluster with the new attribute.

    After you go through this list of attributes, you now have a list of clusters. Then, go through the clusters you created and pick the most representative attribute out of each cluster. Return the list of clusters and representatives in the following JSON format: note that you have to return both the indices and the corresponding attribute.

    ```json
    [
        {{
            "representative": {{
                "index": ...,
                "attribute": ...
            }},
            "members": [
                {{
                    "index": ...,
                    "attribute": ...
                }},
                {{
                    "index": ...,
                    "attribute": ...
                }},
                ...
            ],
        }},
    ]
    ```

    Remember to include the surrounding JSON tags.

    Now, here is the full list of attributes:

    {attributes}
""").strip()


STRICT_CLUSTER_PROMPT = dedent("""
    You will be given a list of textual attribute descriptions.

    Your task is to cluster these attributes into tight clusters. To do this, go through the list of attributes from top to bottom, while maintaining a running list of clusters.

    - Check if the new attribute is semantically similar to any previous cluster. Attributes should be clustered together ONLY if they would manifest almost identically in actual responses (i.e., a response carrying one attribute would almost certainly also exhibit the others in the cluster). If two attributes share some conceptual similarity but could be realized differently in practice, do not put them in the same cluster.

    - If the new attribute is not semantically similar to any previous cluster, create a new cluster with the new attribute as the representative.

    After you go through this list of attributes, you now have a list of clusters. Return the list of clusters in the following JSON format: note that you have to return both the indices and the corresponding attribute.

    ```json
    [
        {{
            "members": [
                {{
                    "index": ...,
                    "attribute": ...
                }},
                ...
            ],
        }},
        ...
    ]
    ```

    Remember to include the surrounding JSON tags.

    Here is the full list of attributes:

    {attributes}
""").strip()


@dataclass
class _UnionFind:
    parent: list[int]
    size: list[int]

    @classmethod
    def create(cls, n: int) -> "_UnionFind":
        return cls(parent=list(range(n)), size=[1] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return groups


class ClusterModel(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    async def cluster_plans(self, to_write: dict[int, list[dict[str, Any]]], n_pop: int, **kwargs) -> dict[int, list[dict[str, Any]]]:
        pass

    @abstractmethod
    async def cluster_by_similarity(
        self,
        inputs: list[str],
        *,
        cosine_sim_threshold: float,
        **kwargs,
    ) -> tuple[dict[int, list[str]], dict[int, list[int]]]:
        """
        Cluster inputs by similarity.
        Returns (niches, indices) where:
        - niches: dict mapping cluster_label -> list of input strings
        - indices: dict mapping cluster_label -> list of input indices
        - cluster_label -1 indicates noise/outliers
        """
        pass


class EmbedClusterModel(ClusterModel):
    def __init__(
        self,
        embed_model_name: str,
        embed_dim: int = -1,
        random_state: int = 42,
    ):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embed_model_name = embed_model_name
        self.embed_dim = embed_dim
        self.random_state = random_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "embed_model_name": self.embed_model_name,
            "embed_dim": self.embed_dim,
            "random_state": self.random_state,
        }

    def embed(self, inputs: list[str], *, embed_dim: int | None = None) -> np.ndarray:
        """Returns L2-normalized embs [n_inputs, embed_dim]"""
        dim = self.embed_dim if embed_dim is None else embed_dim
        embs: np.ndarray = self.embed_model.encode(inputs)  # type: ignore

        if dim != -1:
            embs = embs[:, :dim]

        embs = embs.astype(np.float32, copy=False)
        norms = np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms

        logger.info(f"Embedded shape: {embs.shape}")
        return embs

    def pca(self, embs: np.ndarray, pca_dim: int) -> np.ndarray:
        n_components = min(pca_dim, embs.shape[0], embs.shape[1])
        if n_components >= embs.shape[1]:
            return embs

        pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced = pca.fit_transform(embs).astype(np.float32, copy=False)

        # Re-normalize so dot-product remains cosine similarity.
        norms = np.linalg.norm(reduced, ord=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        reduced = reduced / norms

        logger.info(f"PCA reduced shape: {reduced.shape}")
        return reduced

    def _cosine_stats(self, embs: np.ndarray) -> None:
        # embs are normalized => cosine distance = 1 - dot
        sim = embs @ embs.T
        dist = 1.0 - sim
        upper = dist[np.triu_indices_from(dist, k=1)]
        if upper.size == 0:
            return
        percentiles = list(range(10, 101, 10))
        msg = " | ".join(f"p{p}={np.percentile(upper, p):.3f}" for p in percentiles)
        logger.info(f"Cosine distance distribution: {msg}")

    def _connected_components_by_similarity(
        self,
        embs: np.ndarray,
        *,
        cosine_sim_threshold: float,
    ) -> list[list[int]]:
        """
        Builds connected components where an edge exists if cosine_sim >= threshold.
        embs must be L2-normalized.
        """
        n = embs.shape[0]
        if n == 0:
            return []

        sim = embs @ embs.T  # [n, n], cosine similarity
        uf = _UnionFind.create(n)

        rows, cols = np.where(np.triu(sim, k=1) >= cosine_sim_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            uf.union(i, j)

        comps = list(uf.components().values())
        comps.sort(key=len, reverse=True)
        return comps

    def _medoid_index(self, embs: np.ndarray, indices: list[int]) -> int:
        """Return original index of the cosine-distance medoid for the given subset."""
        if len(indices) == 1:
            return indices[0]

        sub = embs[indices]  # normalized
        sim = sub @ sub.T
        dist = 1.0 - sim
        sum_dist = dist.sum(axis=1)
        best = int(np.argmin(sum_dist))
        return indices[best]

    def pick_representatives(
        self,
        inputs: list[str],
        *,
        n_pop: int,
        cosine_sim_threshold: float,
    ) -> list[dict[str, Any]]:
        """
        1) collapse near-duplicates by cosine threshold (connected components)
        2) if still too many groups, KMeans them down to n_pop
        Returns cluster-like dicts, with center_idx always a real input index.
        """
        if not inputs:
            return []

        embs = self.embed(inputs)
        self._cosine_stats(embs)

        # Stage A: near-duplicate collapse
        comps = self._connected_components_by_similarity(embs, cosine_sim_threshold=cosine_sim_threshold)

        comp_infos: list[dict[str, Any]] = []
        comp_rep_indices: list[int] = []
        comp_sizes: list[int] = []
        for comp in comps:
            rep_idx = self._medoid_index(embs, comp)
            comp_infos.append({"members": comp, "rep_idx": rep_idx})
            comp_rep_indices.append(rep_idx)
            comp_sizes.append(len(comp))

        logger.info(
            f"Dedupe components: {len(comps)} from {len(inputs)} inputs "
            f"(threshold={cosine_sim_threshold:.3f})"
        )

        # print out the deduped entries
        for i, comp in enumerate(comps):
            if len(comp) >= 2:
                logger.info(f"Deduped cluster {i}:\n{"\n".join(inputs[c] for c in comp)}")

        # If we already have <= target groups, return them (can't manufacture more distinct reps).
        if len(comp_infos) <= n_pop:
            results: list[dict[str, Any]] = []
            for cluster_idx, info in enumerate(comp_infos):
                members = sorted(info["members"])
                center_idx = int(info["rep_idx"])
                results.append(
                    {
                        "cluster_idx": cluster_idx,
                        "center_idx": center_idx,
                        "center_input": inputs[center_idx],
                        "content_indices": members,
                    }
                )
            return results

        # Stage B: cluster component representatives down to fixed count
        rep_embs = embs[comp_rep_indices]  # normalized
        k = min(n_pop, rep_embs.shape[0])

        logger.info(
            f"Fitting KMeans over {len(comp_infos)} deduped groups into {k} representatives"
        )
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        kmeans.fit(rep_embs)

        # Group component indices by kmeans cluster
        group_to_comp_idxs: dict[int, list[int]] = defaultdict(list)
        for comp_i, lbl in enumerate(kmeans.labels_.tolist()):  # type: ignore
            group_to_comp_idxs[int(lbl)].append(comp_i)

        results = []
        for cluster_idx in range(k):
            comp_idxs = group_to_comp_idxs.get(cluster_idx, [])
            if not comp_idxs:
                continue

            # Choose weighted medoid among component reps (weights = component sizes)
            sub_embs = rep_embs[comp_idxs]  # normalized
            sim = sub_embs @ sub_embs.T
            dist = 1.0 - sim  # [m, m]
            weights = np.array([comp_sizes[i] for i in comp_idxs], dtype=np.float32)  # [m]

            # For each candidate j: sum_i w_i * dist[j, i]
            weighted_sum_dist = dist @ weights
            best_local = int(np.argmin(weighted_sum_dist))
            best_comp_i = comp_idxs[best_local]
            center_idx = int(comp_infos[best_comp_i]["rep_idx"])

            # Union original indices across all components in this kmeans cluster
            members: list[int] = []
            for ci in comp_idxs:
                members.extend(comp_infos[ci]["members"])
            members = sorted(set(members))

            logger.info(
                f"KMeans cluster {cluster_idx} with {len(members)} members:\n"
                f"Center: {inputs[center_idx]}\n"
                f"Members:\n{"\n".join(inputs[m] for m in members)}"
            )

            results.append(
                {
                    "cluster_idx": cluster_idx,
                    "center_idx": center_idx,
                    "center_input": inputs[center_idx],
                    "content_indices": members,
                }
            )

        for r in results:
            assert r["center_idx"] in r["content_indices"]

        return results
    
    async def cluster_plans(
        self,
        to_write: dict[int, list[dict[str, Any]]],
        n_pop: int,
        *,
        cosine_sim_threshold: float=0.9,
        **kwargs,  # for Liskov
    ) -> dict[int, list[dict[str, Any]]]:
        """
        Cluster plans for each seed into n_pop clusters.
        Returns the representative (medoid) of each cluster.
        """

        to_write_new = defaultdict(list)

        for seed_idx, seed_plans in to_write.items():
            print(f"Clustering {len(seed_plans)} bias candidates for seed {seed_idx}")
            if not seed_plans:
                continue

            all_plans = [plan["plan"] for plan in seed_plans]
            cluster_results = self.pick_representatives(
                inputs=all_plans,
                n_pop=n_pop,
                cosine_sim_threshold=cosine_sim_threshold,
            )

            for result in cluster_results:
                to_write_new[seed_idx].append(
                    {
                        "plan": result["center_input"],
                        "meta": seed_plans[result["center_idx"]]["meta"],
                    }
                )

        return dict(to_write_new)

    async def cluster_by_similarity(
        self,
        inputs: list[str],
        *,
        cosine_sim_threshold: float,
        min_cluster_size: int = 2,
        embed_dim: int | None = None,
        pca_dim: int | None = None,
        **kwargs,
    ) -> tuple[dict[int, list[str]], dict[int, list[int]]]:
        """
        If min_cluster_size >= 2, singletons are labeled as -1 (noise).
        """
        if not inputs:
            return {}, {}

        embs = self.embed(inputs, embed_dim=embed_dim)
        if pca_dim is not None:
            embs = self.pca(embs, pca_dim=pca_dim)

        self._cosine_stats(embs)

        comps = self._connected_components_by_similarity(embs, cosine_sim_threshold=cosine_sim_threshold)

        niches: dict[int, list[str]] = defaultdict(list)
        indices: dict[int, list[int]] = defaultdict(list)

        next_label = 0
        for comp in comps:
            if len(comp) < min_cluster_size:
                for i in comp:
                    niches[-1].append(inputs[i])
                    indices[-1].append(i)
                continue

            lbl = next_label
            next_label += 1
            for i in comp:
                niches[lbl].append(inputs[i])
                indices[lbl].append(i)

        # Logging
        for label, members in sorted(niches.items(), key=lambda kv: kv[0]):
            title = "Noise points" if label == -1 else f"Cluster {label}"
            logger.info(
                f"{title} ({len(members)} items):\n" + "\n".join(f"  - {m}" for m in members)
            )

        return niches, indices


class LLMClusterModel(ClusterModel):
    def __init__(
        self,
        model_name: str = "openai/gpt-5.2",
        max_tokens: int = 50000,
        reasoning: int | str = "high",
        max_par: int = 64,
        force_caller: str | None = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.max_par = max_par

        self.caller = AutoCaller(
            dotenv_path=".env",
            retry_config=RETRY_CONFIG,
            force_caller=force_caller,
        )
        self.force_caller = force_caller

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "max_par": self.max_par,
            "force_caller": self.force_caller,
        }

    async def cluster_plans(self, to_write: dict[int, list[dict[str, Any]]], n_pop: int, **kwargs) -> dict[int, list[dict[str, Any]]]:
        prompts_to_send = []
        seed_indices = []
        seed_attributes = []
        
        for seed_idx, seed_plans in to_write.items():
            seed_indices.append(seed_idx)

            attributes = [plan["plan"] for plan in seed_plans]
            seed_attributes.append(attributes)

            prompts_to_send.append(CLUSTER_PROMPT.format(
                cluster_summary=seed_plans[0]["meta"]["cluster_summary"],
                attributes=json.dumps([{"index": i, "attribute": attr} for i, attr in enumerate(attributes)]),
            ))

        responses = await self.caller.call(
            messages=prompts_to_send,
            model=self.model_name,
            max_parallel=self.max_par,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            enable_cache=True,
            desc="Clustering attributes",
        )

        to_write_new = defaultdict(list)

        for i in range(len(seed_indices)):
            seed_idx = seed_indices[i]
            attributes = seed_attributes[i]
            resp = responses[i]

            if resp is None:
                raise ValueError(f"LLMClusterModel responded None for seed {seed_idx}.")

            cluster_results, reasoning = parse_json_response(resp)
            logger.info(f"Seed {seed_idx} cluster model reasoning:\n{reasoning}")
            logger.info(f"Seed {seed_idx} clustered into {len(cluster_results)} clusters. Taking the most populous {n_pop} clusters.")
            logger.info(f"Seed {seed_idx} cluster results: {json.dumps(cluster_results, indent=4)}")

            try:
                cluster_results.sort(key=lambda x: len(x["members"]), reverse=True)
            except KeyError:
                logger.error(f"Seed {seed_idx} cluster result missing 'members' key: {json.dumps(cluster_results, indent=4)}")
                raise

            for cluster in cluster_results[:n_pop]:  # take the most populous n_pop clusters, if more than that
                rep = cluster["representative"]
                rep_idx = rep["index"]
                rep_attr = rep["attribute"]

                if attributes[rep_idx].strip() != rep_attr.strip():
                    logger.error(f"Index-attribute mismatch for seed {seed_idx}:\nindex: {rep_idx}\nrepresentative: {rep_attr}\nattribute: {attributes[rep_idx]}")
                
                to_write_new[seed_idx].append(
                    {
                        "plan": attributes[rep_idx],
                        "meta": to_write[seed_idx][rep_idx]["meta"],
                    }
                )

        return dict(to_write_new)

    async def cluster_by_similarity(
        self,
        inputs: list[str],
        *,
        cosine_sim_threshold: float,  # Unused for LLM, but kept for interface compatibility
        **kwargs,
    ) -> tuple[dict[int, list[str]], dict[int, list[int]]]:
        """
        Cluster inputs using LLM with strict similarity criteria.
        Note: cosine_sim_threshold is unused for LLM-based clustering.
        """
        if not inputs:
            return {}, {}

        # Create prompt
        prompt = STRICT_CLUSTER_PROMPT.format(
            attributes=json.dumps([{"index": i, "attribute": attr} for i, attr in enumerate(inputs)]),
        )

        # Call LLM
        responses = await self.caller.call(
            messages=[prompt],
            model=self.model_name,
            max_parallel=1,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            enable_cache=False,
            desc="Clustering attributes"
        )
        response = responses[0]

        if response is None:
            raise ValueError("LLMClusterModel responded None for cluster_by_similarity.")

        cluster_results, reasoning = parse_json_response(response)
        logger.info(f"Cluster model reasoning:\n{reasoning}")
        logger.info(f"Clustered into {len(cluster_results)} clusters.")
        logger.info(f"Cluster results: {json.dumps(cluster_results, indent=4)}")

        # Convert to (niches, indices) format
        niches: dict[int, list[str]] = defaultdict(list)
        indices: dict[int, list[int]] = defaultdict(list)

        for cluster_idx, cluster in enumerate(cluster_results):
            try:
                members = cluster["members"]
            except KeyError:
                logger.error(f"Cluster {cluster_idx} missing 'members' key: {json.dumps(cluster, indent=4)}")
                raise

            for member in members:
                member_idx = member["index"]
                niches[cluster_idx].append(inputs[member_idx])
                indices[cluster_idx].append(member_idx)

        # Logging
        for label, members in sorted(niches.items(), key=lambda kv: kv[0]):
            logger.info(
                f"Cluster {label} ({len(members)} items):\n" + "\n".join(f"  - {m}" for m in members)
            )

        return niches, indices