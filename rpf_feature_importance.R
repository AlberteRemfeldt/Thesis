

rpf_feature_importance <- function(res) {
  fam_importance <- lapply(1:ncol(res), function(s) unlist(res[,s][[9]]) /sum(unlist(res[,s][[9]])))
  total_importance <- rowMeans(matrix(unlist(fam_importance), ncol =length(fam_importance)))
  return (total_importance)
}

