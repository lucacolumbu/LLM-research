# Literature pass 2 (2026-09-13), for paper 1

Compiled by a search agent; V = page opened and abstract confirmed, U = search snippet only. Full entry list with links is in the session record; this file keeps the entries that shape the paper.

## Closest prior work to each claim

**A. Gzip gain predicts induction formation; fails under vocabulary shift; tier 2 repairs it.**
- Aoyama, Wilcox, Schneider, "Predicting the Emergence of Induction Heads in Language Model Pretraining", ICML 2026, arXiv 2511.16893 (V). Formation predicted from batch size x context size; surface bigram repetition frequency and reliability set whether heads form. Closest: statistic-specific predictor, no compressor, no failure regime, no repair.
- Pandey, "gzip Predicts Data-dependent Scaling Laws", 2024, arXiv 2405.16684 (V). gzip compressibility predicts shifts of the compute-optimal frontier on PCFG data. Loss-level analogue.
- Chen, Luo, Pan, "Mechanistic Data Attribution", ICML 2026, arXiv 2601.21996 (V). Influence functions attribute induction heads to training samples in Pythia; repetitive structured data catalyses them; an augmentation pipeline speeds formation. Post hoc and model-dependent where ours is a priori and model-free.
- Deletang et al., "Language Modeling Is Compression", ICLR 2024, arXiv 2309.10668 (V): the tier-1/tier-2 equivalence.
- Baronchelli, Caglioti, Loreto, "Measuring complexity with zippers", arXiv physics/0605031 (V): the tier-1 estimator.

**B. Gzip-selected subset forms the head ~7x sooner than random.**
- Chen, Luo, Pan 2026 (above); Wang and Murfet, "Patterning: The Dual of Interpretability", 2026, arXiv 2601.13548 (V): data reweighting via susceptibilities to accelerate or delay induction formation, model-dependent. ZIP-FIT, arXiv 2410.18194 (V): gzip alignment to a target set, loss endpoint. Sabry and Belz, "Induction Signatures Are Not Enough", arXiv 2509.22947 (V): synthetic copy insertion raises induction activity but yields less load-bearing circuits; makes faithfulness a required companion metric.
- No prior work selects by a model-free per-document compression score with formation step as the endpoint.

**C. Lag-k induction circuits; k chosen by data and architecture; lag caps accuracy.**
- Akyurek, Wang, Kim, Andreas, "In-Context Language Learning: Architectures and Algorithms", ICML 2024, arXiv 2401.12973 (V): n-gram heads widen the match key, not the copy offset.
- Chen, Sheen, Wang, Yang, "Unveiling Induction Heads", 2024, arXiv 2409.10559 (V): generalized induction head copying a window with FFN-selected parents; no discrete lag or ceiling.
- Wang and Sato, "Rethinking Associative Memory Mechanism in Induction Head", COLM 2025, arXiv 2412.11459 (V): relative PE makes the previous-token head position-independent; matches our rotary/learned-position finding. Barbero et al., "Round and Round We Go", ICLR 2025, arXiv 2410.06205 (U): RoPE high frequencies build positional patterns.
- Feucht, Todd, Wallace, Bau, "The Dual-Route Model of Induction", COLM 2025, arXiv 2504.03022 (V): end-of-word offsets, a semantically defined lag. Bajaj et al. 2026, arXiv 2604.01094 (V): names the "+1 lag" and treats it as fixed.
- Singh, Moskovitz, Hill, Chan, Saxe, "What needs to go right for an induction head?", ICML 2024, arXiv 2404.07129 (V): three subcircuits with data-dependent timing; the decomposition needed to locate the lag.
- Varre, Yuce, Flammarion, ICML 2025, arXiv 2508.12837 (V): sub-n-gram solutions are near-stationary, explaining plateaus. Wang et al., rLLC head specialisation, arXiv 2410.02984 (V): multigram circuits in 2-layer attention-only models.
- Lag as a learned parameter with an accuracy ceiling is not described anywhere found.

**D. IOI solved by S-inhibition plus MLP promotion, no name mover; the prior lives in MLP 0.**
- Wang et al., IOI, ICLR 2023, arXiv 2211.00593 (V). McDougall et al., "Copy Suppression", arXiv 2310.04625 (V): an exclusion mechanism in GPT-2. Adhikari, "Emergence of Minimal Circuits for IOI in Attention-Only Transformers", ACL 2026 SRW, arXiv 2510.25013 (V): from-scratch attention-only IOI circuits differ from GPT-2's; no MLPs, no held-out-name analysis. Miller and Neo, Alignment Forum 2023 (V, footnote quoting Nanda): MLP 0 as extended embedding.
- Merullo, Eickhoff, Pavlick, ICLR 2024, arXiv 2310.08744 (V): circuit reuse across tasks (the pretrained analogue of our interference plan). Tigges et al., NeurIPS 2024, arXiv 2407.10827 (V): IOI circuits consistent across training and scale.

## Results most likely to complicate A/B
- Lee, Smith, Adam, Hoogland, "Influence Dynamics and Stagewise Data Attribution", arXiv 2510.12071 (V): influence flips sign at developmental transitions, so a fixed per-document score is not stage-invariant. Our late-leak result is an instance.
- Singh et al., "The Transient Nature of Emergent ICL", NeurIPS 2023, arXiv 2311.08360 (V): ICL can appear then vanish; track persistence, not only formation.
- Hernandez et al., "Scaling Laws and Interpretability of Learning from Repeated Data", arXiv 2205.10487 (V): corpus-level duplication damages induction heads; within-context repetition helps. Say which one the zipper measures (within-document).

## Also relevant
Olsson et al. 2022 (induction heads, V); Chan et al., NeurIPS 2022, arXiv 2205.05055 (V); Reddy, ICLR 2024, arXiv 2312.03002 (V); Bietti et al., NeurIPS 2023, arXiv 2306.00802 (V); Edelman et al., NeurIPS 2024, arXiv 2402.11004 (V); D'Angelo et al., ICML 2026, arXiv 2607.02800 (V); Musat et al., arXiv 2511.01033 (V); Sahin et al., "In-Context Learning Without Copying", arXiv 2511.05743 (V); Minegishi et al., ICML 2025, arXiv 2505.16694 (V); Gibson, Cui, Reddy 2026, arXiv 2604.12151 (V); Wang et al., "Embryology of a Language Model", arXiv 2508.00331 (V); Venkatesh 2026, arXiv 2605.08853 (V); DSIR arXiv 2302.03169 (V); RHO-LOSS arXiv 2206.07137 (V); DoReMi (U); Ankner et al. perplexity pruning arXiv 2405.20541 (U); Lee et al. dedup ACL 2022 (V); Jiang et al. gzip classification ACL 2023 (V); Elhage et al. 2021 (V); Chen et al. "Sudden Drops in the Loss", ICLR 2024 (V).
