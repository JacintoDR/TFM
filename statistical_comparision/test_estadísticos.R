## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
library("scmamp")


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
resultados <- read.csv("test_res.csv", header = TRUE, sep = '\t')
tablatst <- cbind(resultados[,1:dim(resultados)[2]])
colnames(tablatst) <- names(resultados)[1:dim(resultados)[2]]


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
colnames(tablatst)

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
tablatst


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Test de Fligner-Killeen para la homocedasticidad (no paramétrico, mediana, no normalidad)
fligner.test(x = list(tablatst$SMOTE, tablatst$ROS, tablatst$VAE, tablatst$GAN))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Aplicación del test de Friedman Aligned Ranks
test_friedman_aligned_ranks <- friedmanAlignedRanksTest(as.matrix(tablatst))
test_friedman_aligned_ranks


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Aplicación del test de Friedman Aligned Ranks post-hoc
test_friedman_aligned_ranks_post <- friedmanAlignedRanksPost(as.matrix(tablatst))
test_friedman_aligned_ranks_post


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Bergmann and Hommel dynamic correction of p-values
adjustBergmannHommel (test_friedman_aligned_ranks_post)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
quadeTest(as.matrix(tablatst))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
quadePost(as.matrix(tablatst))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
plotDensities(as.matrix(tablatst))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
test <- postHocTest(as.matrix(tablatst), test="friedman", correct="bergmann", use.rank=TRUE)
plotRanking(pvalues=test$corrected.pval, summary=test$summary, alpha=0.05)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
##TABLA NORMALIZADA - smote (other) vs vae (ref) para WILCOXON
# + 0.1 porque wilcox R falla para valores == 0 en la tabla

difs <- (tablatst[,1] - tablatst[,3]) / tablatst[,1]
wilc_1_2 <- cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1), ifelse (difs>0, 	abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) <- c(colnames(tablatst)[1], colnames(tablatst)[3])
head(wilc_1_2)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Aplicación del test de WILCOXON
SMvsVAEtst <- wilcox.test(wilc_1_2[,1], wilc_1_2[,2], alternative = "two.sided", paired=TRUE)
Rmas <- SMvsVAEtst$statistic
pvalue <- SMvsVAEtst$p.value
SMvsVAEtst <- wilcox.test(wilc_1_2[,2], wilc_1_2[,1], alternative = "two.sided", paired=TRUE)
Rmenos <- SMvsVAEtst$statistic
Rmas
Rmenos
pvalue



## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
##TABLA NORMALIZADA - smote (other) vs gan (ref) para WILCOXON
# + 0.1 porque wilcox R falla para valores == 0 en la tabla

difs <- (tablatst[,1] - tablatst[,4]) / tablatst[,1]
wilc_1_2 <- cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1), ifelse (difs>0, 	abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) <- c(colnames(tablatst)[1], colnames(tablatst)[4])
head(wilc_1_2)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Aplicación del test de WILCOXON
SMvsGANtst <- wilcox.test(wilc_1_2[,1], wilc_1_2[,2], alternative = "two.sided", paired=TRUE)
Rmas <- SMvsGANtst$statistic
pvalue <- SMvsGANtst$p.value
SMvsGANtst <- wilcox.test(wilc_1_2[,2], wilc_1_2[,1], alternative = "two.sided", paired=TRUE)
Rmenos <- SMvsGANtst$statistic
Rmas
Rmenos
pvalue



## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
##TABLA NORMALIZADA - vae (other) vs gan (ref) para WILCOXON
# + 0.1 porque wilcox R falla para valores == 0 en la tabla

difs <- (tablatst[,3] - tablatst[,4]) / tablatst[,3]
wilc_1_2 <- cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1), ifelse (difs>0, 	abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) <- c(colnames(tablatst)[3], colnames(tablatst)[4])
head(wilc_1_2)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Aplicación del test de WILCOXON
VAEvsGANtst <- wilcox.test(wilc_1_2[,1], wilc_1_2[,2], alternative = "two.sided", paired=TRUE)
Rmas <- VAEvsGANtst$statistic
pvalue <- VAEvsGANtst$p.value
VAEvsGANtst <- wilcox.test(wilc_1_2[,2], wilc_1_2[,1], alternative = "two.sided", paired=TRUE)
Rmenos <- VAEvsGANtst$statistic
Rmas
Rmenos
pvalue


