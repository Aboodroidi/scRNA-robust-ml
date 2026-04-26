**Table B4** Paired bootstrap and Stuart-Maxwell test results comparing SANN_PCA against four established cell-type classification tools on the 8K and 3K cross-donor test sets. Δ macro-F1 is computed as SANN minus baseline on each of 1,000 paired bootstrap resamples of the test cells; mean and 95% percentile CI reported. Stuart-Maxwell test (multi-class extension of McNemar) compares paired marginal prediction distributions on the same test cells. Positive Δ favours SANN_PCA; CIs excluding zero indicate the margin is robust to test-set sampling variation. **Bold** marks comparisons where the 95% CI excludes zero. scANVI was evaluated under both its default configuration and the inverse-frequency class-weighted variant introduced in §4.1.3.

| Comparison | Test donor | n cells | Δ macro-F1 (mean) | 95% CI | McNemar p |
|---|---|---|---|---|---|
| SANN vs Seurat | 8K | 6,534 | -0.0028 | [-0.0069, +0.0014] | <0.001 |
| SANN vs Seurat | 3K | 2,601 | +0.0078 | [-0.0032, +0.0289] | 0.227 |
| SANN vs ACTINN | 8K | 6,534 | **+0.0264** | **[+0.0192, +0.0339]** | 0.002 |
| SANN vs ACTINN | 3K | 2,601 | **+0.0299** | **[+0.0155, +0.0534]** | 0.040 |
| SANN vs SingleR | 8K | 6,534 | **+0.1060** | **[+0.0977, +0.1138]** | <0.001 |
| SANN vs SingleR | 3K | 2,601 | **+0.3361** | **[+0.3162, +0.3507]** | 0.109 |
| SANN vs scANVI (default) | 8K | 6,534 | **+0.0898** | **[+0.0803, +0.0997]** | <0.001 |
| SANN vs scANVI (default) | 3K | 2,601 | **+0.1291** | **[+0.0842, +0.1882]** | <0.001 |
| SANN vs scANVI (weighted, seed 42) | 8K | 6,534 | **+0.1810** | **[+0.1704, +0.1918]** | <0.001 |
| SANN vs scANVI (weighted, seed 42) | 3K | 2,601 | **+0.0974** | **[+0.0804, +0.1212]** | <0.001 |
