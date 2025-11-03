# Helper script to export vegan reference values for nuee comparison.
#
# Usage (from repo root in R):
#   source("tests/reference/export_vegan.R")
#
# The script writes a single RDS/JSON pair that captures representative
# outputs for the vegan functions mirrored in nuee. These payloads are
# consumed by the Python diagnostics to assert numerical compatibility.

library(vegan)
library(jsonlite)

setwd("C:/Users/parse01/documents-locaux/GitHub/nuee")

set.seed(123)

data(varespec)
data(varechem)

env_vars <- c("N", "P", "K", "Ca", "Mg")
grouping <- factor(ifelse(varechem$Humdepth > median(varechem$Humdepth),
                          "High", "Low"),
                   levels = c("Low", "High"))

matrix_payload <- function(mat) {
  if (is.null(mat)) {
    return(NULL)
  }
  m <- as.matrix(mat)
  list(
    data = unname(m),
    rownames = if (!is.null(rownames(m))) as.character(rownames(m)) else NULL,
    colnames = if (!is.null(colnames(m))) as.character(colnames(m)) else NULL
  )
}

vector_payload <- function(vec) {
  if (is.null(vec)) {
    return(NULL)
  }
  vals <- unname(as.numeric(vec))
  list(
    data = vals,
    names = if (!is.null(names(vec))) as.character(names(vec)) else NULL
  )
}

dist_payload <- function(d) {
  if (is.null(d)) {
    return(NULL)
  }
  list(
    data = unname(as.numeric(d)),
    size = attr(d, "Size"),
    labels = if (!is.null(attr(d, "Labels"))) as.character(attr(d, "Labels")) else NULL,
    method = attr(d, "method")
  )
}

table_payload <- function(tbl) {
  if (is.null(tbl)) {
    return(NULL)
  }
  df <- as.data.frame(tbl)
  matrix_payload(df)
}

# Ordination -------------------------------------------------------------------

meta <- vegan::metaMDS(varespec, k = 2, distance = "bray",
                       trymax = 20, trace = FALSE)
rda_model <- vegan::rda(varespec, varechem[, env_vars])
cca_model <- vegan::cca(varespec, varechem[, env_vars])
pca_model <- stats::prcomp(varespec, center = TRUE, scale. = TRUE)
envfit_model <- vegan::envfit(meta, varechem[, env_vars], permutations = 199)
rda_sites_scaling1 <- scores(rda_model, display = "sites", scaling = 1)
rda_sites_for_procrustes <- rda_sites_scaling1[, seq_len(ncol(meta$points)), drop = FALSE]
procrustes_model <- vegan::procrustes(rda_sites_for_procrustes,
                                      meta$points, symmetric = TRUE)
procrustes_summary <- summary(procrustes_model)

ordination_reference <- list(
  metaMDS = list(
    stress = meta$stress,
    trymax = meta$trymax,
    points = matrix_payload(meta$points),
    species = matrix_payload(meta$species)
  ),
  rda = list(
    eigenvalues = list(
      constrained = unname(rda_model$CCA$eig),
      unconstrained = unname(rda_model$CA$eig)
    ),
    bases = list(
      cca_wa = matrix_payload(rda_model$CCA$wa),
      ca_u = matrix_payload(rda_model$CA$u),
      cca_v = matrix_payload(rda_model$CCA$v),
      ca_v = matrix_payload(rda_model$CA$v)
    ),
    weights = list(
      row_sums = unname(rda_model$rowsum),
      col_sums = unname(rda_model$colsum)
    ),
    totals = list(
      tot_chi = rda_model$tot.chi
    ),
    scores = list(
      scaling1 = list(
        sites = matrix_payload(vegan::scores(rda_model, display = "sites", scaling = 1)),
        species = matrix_payload(vegan::scores(rda_model, display = "species", scaling = 1))
      ),
      scaling2 = list(
        sites = matrix_payload(vegan::scores(rda_model, display = "sites", scaling = 2)),
        species = matrix_payload(vegan::scores(rda_model, display = "species", scaling = 2))
      ),
      scaling3 = list(
        sites = matrix_payload(vegan::scores(rda_model, display = "sites", scaling = 3)),
        species = matrix_payload(vegan::scores(rda_model, display = "species", scaling = 3))
      )
    )
  ),
  cca = list(
    eigenvalues = list(
      constrained = unname(cca_model$CCA$eig),
      unconstrained = unname(cca_model$CA$eig)
    ),
    bases = list(
      cca_u = matrix_payload(cca_model$CCA$u),
      cca_v = matrix_payload(cca_model$CCA$v),
      cca_wa = matrix_payload(cca_model$CCA$wa),
      ca_u = matrix_payload(cca_model$CA$u),
      ca_v = matrix_payload(cca_model$CA$v)
    ),
    totals = list(
      tot_chi = cca_model$tot.chi
    ),
    scores = list(
      scaling1 = list(
        sites = matrix_payload(vegan::scores(cca_model, display = "sites", scaling = 1)),
        species = matrix_payload(vegan::scores(cca_model, display = "species", scaling = 1))
      ),
      scaling2 = list(
        sites = matrix_payload(vegan::scores(cca_model, display = "sites", scaling = 2)),
        species = matrix_payload(vegan::scores(cca_model, display = "species", scaling = 2))
      ),
      scaling3 = list(
        sites = matrix_payload(vegan::scores(cca_model, display = "sites", scaling = 3)),
        species = matrix_payload(vegan::scores(cca_model, display = "species", scaling = 3))
      )
    )
  ),
  pca = list(
    sdev = unname(pca_model$sdev),
    rotation = matrix_payload(pca_model$rotation),
    x = matrix_payload(pca_model$x),
    center = unname(pca_model$center),
    scale = unname(pca_model$scale)
  ),
  envfit = list(
    r = vector_payload(envfit_model$vectors$r),
    r2 = vector_payload(envfit_model$vectors$r^2),
    pvals = vector_payload(envfit_model$vectors$pvals),
    scores = matrix_payload(scores(envfit_model, "vectors"))
  ),
  procrustes = list(
    correlation = unname(procrustes_summary$correlation),
    ss = unname(procrustes_model$ss),
    rotation = matrix_payload(procrustes_model$rotation),
    translation = vector_payload(procrustes_model$translation)
  )
)

# Dissimilarity & permutation ---------------------------------------------------

vegdist_bray <- vegan::vegdist(varespec, method = "bray")
adonis2_model <- vegan::adonis2(varespec ~ N + P + K + Ca + Mg,
                                data = varechem, permutations = 199)
anosim_model <- vegan::anosim(vegdist_bray, grouping, permutations = 199)
mrpp_model <- vegan::mrpp(varespec, grouping, permutations = 199)
betadisper_model <- vegan::betadisper(vegdist_bray, grouping, type = "centroid")
permutest_betadisper <- vegan::permutest(betadisper_model, permutations = 199)
mantel_model <- vegan::mantel(vegdist_bray, vegdist(varechem[, env_vars]), permutations = 199)
mantel_partial_model <- vegan::mantel.partial(vegdist_bray, vegdist(varechem[, env_vars]), vegdist(varechem[, env_vars[-1]]), permutations = 199)
protest_model <- vegan::protest(meta$points,
                                rda_sites_for_procrustes,
                                permutations = 199)
protest_ss <- as.numeric(protest_model$ss)[1]
protest_correlation <- sqrt(max(0, 1 - protest_ss))
protest_signif <- as.numeric(protest_model$signif)
anova_rda <- stats::anova(rda_model, permutations = 199)
permutest_rda <- vegan::permutest(rda_model, permutations = 199)
permutest_rda_tab <- permutest_rda$tab
if (is.null(permutest_rda_tab) || nrow(permutest_rda_tab) == 0) {
  permutest_rda_tab <- data.frame(
    Df = c(permutest_rda$df, permutest_rda$df.residual),
    Variance = c(permutest_rda$tot.chi, permutest_rda$res.chi),
    F = c(permutest_rda$f, NA_real_),
    `Pr(>F)` = c(permutest_rda$signif, NA_real_)
  )
  rownames(permutest_rda_tab) <- c("Model", "Residual")
}
permutest_rda_permutations <- permutest_rda$permutations
if (is.null(permutest_rda_permutations)) {
  permutest_rda_permutations <- attr(permutest_rda, "permutations")
}
if (is.null(permutest_rda_permutations)) {
  permutest_rda_permutations <- 199
}
if (is.list(permutest_rda_permutations) && !is.null(permutest_rda_permutations$nperm)) {
  permutest_rda_permutations <- permutest_rda_permutations$nperm
}
permutest_rda_permutations <- as.integer(permutest_rda_permutations)

dissimilarity_reference <- list(
  vegdist_bray = dist_payload(vegdist_bray),
  adonis2 = table_payload(adonis2_model),
  anosim = list(
    statistic = anosim_model$statistic,
    signif = anosim_model$signif,
    permutations = anosim_model$permutations,
    dis_rank = vector_payload(anosim_model$dis.rank)
  ),
  mrpp = list(
    delta = mrpp_model$delta,
    e_delta = mrpp_model$E.delta,
    chance_corrected = as.numeric(mrpp_model$chance.corrected),
    p_value = mrpp_model$Pvalue
  ),
  betadisper = list(
    distances = vector_payload(betadisper_model$distances),
    centroids = matrix_payload(betadisper_model$centroids),
    eig = vector_payload(betadisper_model$eig)
  ),
  permutest_betadisper = list(
    tab = table_payload(permutest_betadisper$tab),
    permutations = permutest_betadisper$permutations
  ),
  mantel = list(
    statistic = as.numeric(mantel_model$statistic),
    signif = mantel_model$signif,
    permutations = mantel_model$permutations
  ),
  mantel_partial = list(
    statistic = as.numeric(mantel_partial_model$statistic),
    signif = mantel_partial_model$signif,
    permutations = mantel_partial_model$permutations
  ),
  protest = list(
    correlation = protest_correlation,
    statistic = protest_correlation,
    signif = protest_signif,
    permutations = protest_model$permutations
  ),
  anova_rda = table_payload(anova_rda),
  permutest_rda = list(
    tab = table_payload(permutest_rda_tab),
    permutations = permutest_rda_permutations
  )
)

# Diversity ---------------------------------------------------------------------

specaccum_random <- specaccum(varespec, method = "random", permutations = 50)
poolaccum_model <- poolaccum(varespec, permutations = 50)

shannon_vals <- vegan::diversity(varespec, index = "shannon")
simpson_vals <- vegan::diversity(varespec, index = "simpson")
specnumber_vals <- vegan::specnumber(varespec)
evenness_vals <- shannon_vals / log(pmax(specnumber_vals, 2))
evenness_vals[!is.finite(evenness_vals)] <- NA_real_

diversity_reference <- list(
  shannon = vector_payload(shannon_vals),
  simpson = vector_payload(simpson_vals),
  evenness_shannon = vector_payload(evenness_vals),
  fisher_alpha = vector_payload(vegan::fisher.alpha(round(varespec))),
  renyi = matrix_payload(as.matrix(vegan::renyi(varespec, scales = c(0, 0.5, 1, 2, 3, 4)))),
  rarefy = vector_payload(vegan::rarefy(round(varespec), sample = 20)),
  specnumber = vector_payload(specnumber_vals),
  estimateR = matrix_payload(vegan::estimateR(round(varespec))),
  specaccum = list(
    sites = unname(specaccum_random$sites),
    richness = unname(specaccum_random$richness),
    sd = unname(specaccum_random$sd)
  ),
  poolaccum = list(
    sites = unname(poolaccum_model$Sites),
    richness = unname(poolaccum_model$Richness)
  )
)

# Reference bundle --------------------------------------------------------------

reference <- list(
  datasets = list(
    env_vars = env_vars,
    grouping = as.character(grouping)
  ),
  ordination = ordination_reference,
  dissimilarity = dissimilarity_reference,
  diversity = diversity_reference
)

dir.create("tests/reference", showWarnings = FALSE, recursive = TRUE)
saveRDS(reference, file = "tests/reference/vegan_reference.rds")
write_json(reference,
           path = "tests/reference/vegan_reference.json",
           digits = NA,
           auto_unbox = TRUE,
           pretty = TRUE,
           na = "null")

cat("Wrote vegan reference to tests/reference/vegan_reference.rds\n")
cat("Wrote vegan reference to tests/reference/vegan_reference.json\n")
