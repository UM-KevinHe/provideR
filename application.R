path.input <- "readmission2018.csv"
Rcpp::sourceCpp('library.cpp')

suppressWarnings(data <- data.table::fread(path.input, header=T, 
                                           na.strings=c(".","na",""), sep=",", data.table=F))
data <- data.table::fread(path.input, colClasses=unname(sapply(data,class)),header=T, 
                          na.strings=c(".","na",""), sep=",", data.table=F)
Y.char <- "readmit30_flag"
Z.char <- c("sex","agesp1c60","agesp2c60","agesp3c60","esrdcause_diab",
            "bmi_under","bmi_over","bmi_obese","vincat2","vincat3",
            "vincat4","vincat5","MA_ind","NH_lt90","NH_ge90",
            "ahrq_10", "ahrq_100", "ahrq_101", "ahrq_102", "ahrq_106",
            "ahrq_107", "ahrq_108", "ahrq_117", "ahrq_118", "ahrq_120", 
            "ahrq_121", "ahrq_122", "ahrq_127", "ahrq_130", "ahrq_131", 
            "ahrq_133", "ahrq_134", "ahrq_135", "ahrq_138", "ahrq_140", 
            "ahrq_141", "ahrq_151", "ahrq_152", "ahrq_153", "ahrq_154", 
            "ahrq_155", "ahrq_158", "ahrq_159", "ahrq_197", "ahrq_198", 
            "ahrq_199", "ahrq_201", "ahrq_237", "ahrq_244", "ahrq_251", 
            "ahrq_253", "ahrq_255", "ahrq_259", "ahrq_42", "ahrq_50", 
            "ahrq_51", "ahrq_52", "ahrq_55", "ahrq_59", "ahrq_6",
            "ahrq_64", "ahrq_651", "ahrq_659", "ahrq_660", "ahrq_661", 
            "ahrq_95", "ahrq_96", "ahrq_97","risky_currentdx",
            "timeinhosp_quantile2","timeinhosp_quantile3","timeinhosp_quantile4",
            "diabetes_agesp2c60", "diabetes_agesp3c60")
prov.char <- "prov_ID"
fac.char <- "provfs"
pat.char <- "KECC_ID"
hosp.char <- "prov_hosp"
timetoevents.char <- "days_to_ce"
type.char <- "ce_type"
cutoff <- 10
check <- F

data.prep <- fe.data.prep(data,Y.char,Z.char,fac.char,cutoff,check)
data.prep <- data.prep[data.prep$included==1,]
data.prep <- data.prep[order(data.prep[,fac.char],data.prep[,pat.char]), ]
prop.readm <- sapply(split(data.prep[, Y.char], data.prep[,fac.char]), sum) / 
  sapply(split(data.prep[, Y.char], data.prep[,fac.char]), length)
grp.prop.readm <- cut(prop.readm, 
                      quantile(prop.readm, c(0,0.1,0.9,1)), 
                      labels=c("low", "medium", "high"), T)
n_prov <- sapply(split(data.prep[, Y.char], data.prep[, fac.char]), length)
grp.size <- cut(n_prov, 
                quantile(n_prov, c(0,0.1,0.9,1)), 
                labels=c("small", "medium", "large"), T)
n_prov_pat <- lapply(split(data.prep[, pat.char], data.prep[, fac.char]), 
                     function(x) sapply(split(x, x), length))
gamma <- rep(0, length(unique(data.prep[,fac.char])))
beta <- rep(0, length(Z.char))
eta <- rep(0, length(unique(data.prep[,timetoevents.char]))-1)
fail <- data.prep[,Y.char]
Z <- as.matrix(data.prep[,Z.char])
time <- data.prep[,timetoevents.char]
prov <- data.prep[,fac.char]
# Rcpp
# logit link
system.time(res.Rcpp.logit <- bin_logit_cr(fail, Z, n_prov, time, 
                                           gamma, eta, beta))
# user  system elapsed
# 27.629  11.808  39.512

# constrained model fitting (logit)
m <- length(n_prov)
gamma.med <- median(res.Rcpp.logit$gamma)
cl <- parallel::makeCluster(min(m, parallel::detectCores()))
doParallel::registerDoParallel(cl)
`%dopar%` <- foreach::`%dopar%`
system.time(logit.cstr <- foreach::foreach(i=1:m,.noexport="bin_logit_cr_cstr") %dopar% {
  Rcpp::sourceCpp("library.cpp")
  bin_logit_cr_cstr(fail, Z, n_prov, time, res.Rcpp.logit$gamma, 
                    res.Rcpp.logit$eta, res.Rcpp.logit$beta, 
                    i, gamma.med)
})

# robust score test with model refitting
system.time(score.refit <- foreach::foreach(i=1:m,.combine=cbind,.noexport="bin_logit_cr_score") %dopar% {
  Rcpp::sourceCpp("library.cpp")
  bin_logit_cr_score(fail, Z, n_prov, n_prov_pat, time, 
                     logit.cstr[[i]]$gamma, logit.cstr[[i]]$eta, 
                     logit.cstr[[i]]$beta, i)
})

# robust score test without model refitting
system.time(score.norefit <- foreach::foreach(i=1:m, .combine=cbind, .noexport="bin_logit_cr_score") %dopar% {
  Rcpp::sourceCpp("library.cpp")
  bin_logit_cr_score(fail, Z, n_prov, n_prov_pat, time, 
                     res.Rcpp.logit$gamma, res.Rcpp.logit$eta, 
                     res.Rcpp.logit$beta, i)
})

parallel::stopCluster(cl)

wald.norefit <- bin_logit_cr_wald(fail, Z, n_prov, n_prov_pat, time, 
                                  res.Rcpp.logit$gamma, res.Rcpp.logit$eta, 
                                  res.Rcpp.logit$beta)

upper.panel <- function(x, y, ...) {
  points(x,y, pch=c(15,17,19)[grp.prop.readm], 
         col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red")[grp.prop.readm],
         xlim=c(-4,8), ylim=c(-4,8),...)
  abline(0, 1)
  abline(h=qnorm(alpha/2), lty=2)
  abline(h=-qnorm(alpha/2), lty=2)
  abline(v=qnorm(alpha/2), lty=2)
  abline(v=-qnorm(alpha/2), lty=2)
  r <- round(cor(x, y), digits=2)
  # txt <- paste0("CORR = ", r)
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # text(0.8, 0.1, txt, cex=1.5)
  legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=1.5,
         pch=c(15,17,19), col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red"), 
         bty="n", title="readm rate")
}

lower.panel <- function(x, y, ...) {
  points(x,y, pch=c(15,17,19)[grp.size], col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red")[grp.size],
         xlim=c(-4,8), ylim=c(-4,8),...)
  abline(0, 1)
  abline(h=qnorm(alpha/2), lty=2)
  abline(h=-qnorm(alpha/2), lty=2)
  abline(v=qnorm(alpha/2), lty=2)
  abline(v=-qnorm(alpha/2), lty=2)
  r <- round(cor(x, y), digits=2)
  # txt <- paste0("CORR = ", r)
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # text(0.8, 0.1, txt, cex=1.5)
  legend("topleft", legend=c("smallest 10%", "medium", "largest 10%"), cex=1.5,
         pch=c(15,17,19), col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red"), 
         bty="n", title="discharge ct")
}

panel.hist <- function(x, ...) {
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  h <- hist(x, plot = F, breaks=100)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col="#00AFBB", ...)
  abline(v=qnorm(alpha/2), lty=2)
  abline(v=-qnorm(alpha/2), lty=2)
}

alpha <- 0.05
cex <- 1.1
pdf(paste0("score_tests_3pairs_", Sys.Date(), ".pdf"), height=9, width=9)
data.for.pairs <- t(score.norefit[6:8,])
colnames(data.for.pairs) <- c("stabrobust", "robust", "model")
pairs(data.for.pairs, lower.panel=lower.panel, 
      upper.panel=upper.panel, diag.panel=panel.hist, 
      gap=0.5, oma=c(2,2,2,2), cex.axis=cex)
dev.off()

# score vs Wald tests
pdf(paste0("waldvsscore_scatter_", Sys.Date(), ".pdf"), height=3, width=9)
cex <- 1.5
par(mar=c(3,3,0.5,.5)+0.1, mfrow=c(1,3))
plot(score.norefit[6,], wald.norefit$stat[,6], pch=c(15,17,19)[grp.prop.readm], bty="l", 
     xaxt="n", yaxt="n", ann=F,
     col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red")[grp.prop.readm])
abline(0,1)
axis(1, mgp=c(3,0.6,0))
axis(2, las=1, mgp=c(3,.7,0))
legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=cex, 
       pch=c(15,17,19), col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red"), 
       bty="n", title="readm rate")
mtext("score stat (stabrobust)", side=1, line=1.7)
mtext("Wald stat (stabrobust)", side=2, line=1.6)

plot(score.norefit[7,], wald.norefit$stat[,7], pch=c(15,17,19)[grp.prop.readm], bty="l", 
     xaxt="n", yaxt="n", ann=F,
     col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red")[grp.prop.readm])
abline(0,1)
axis(1, mgp=c(3,0.6,0))
axis(2, las=1, mgp=c(3,.7,0))
legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=cex, 
       pch=c(15,17,19), col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red"), 
       bty="n", title="readm rate")
mtext("score stat (robust)", side=1, line=1.7)
mtext("Wald stat (robust)", side=2, line=2)

plot(score.norefit[8,], wald.norefit$stat[,8], pch=c(15,17,19)[grp.prop.readm], bty="l", 
     xaxt="n", yaxt="n", ann=F,
     col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red")[grp.prop.readm])
abline(0,1)
axis(1, mgp=c(3,0.6,0))
axis(2, las=1, mgp=c(3,.7,0))
legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=cex, 
       pch=c(15,17,19), col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red"), 
       bty="n", title="readm rate")
mtext("score stat (model)", side=1, line=1.7)
mtext("Wald stat (model)", side=2, line=1.6)
dev.off()

# model refitting vs no model refitting
pdf(paste0("score_scatter_refit_vs_norefit", Sys.Date(), ".pdf"), height=3, width=9)
cex <- 1.5
par(mar=c(3,3,0.5,.5)+0.1, mfrow=c(1,3))
plot(score.norefit[6,], score.refit[6,], pch=c(15,17,19)[grp.prop.readm], bty="l", 
     xaxt="n", yaxt="n", ann=F,
     col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red")[grp.prop.readm])
abline(0,1)
axis(1, mgp=c(3,0.6,0))
axis(2, las=1, mgp=c(3,.7,0))
legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=cex, 
       pch=c(15,17,19), col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red"), 
       bty="n", title="readm rate")
mtext("stabrobust w/ model refitting", side=1, line=1.7)
mtext("stabrobust w/o model refitting", side=2, line=1.6)

plot(score.norefit[7,], score.refit[7,], pch=c(15,17,19)[grp.prop.readm], bty="l", 
     xaxt="n", yaxt="n", ann=F,
     col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red")[grp.prop.readm])
abline(0,1)
axis(1, mgp=c(3,0.6,0))
axis(2, las=1, mgp=c(3,.7,0))
legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=cex, 
       pch=c(15,17,19), col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red"), 
       bty="n", title="readm rate")
mtext("robust w/ model refitting", side=1, line=1.7)
mtext("robust w/o model refitting", side=2, line=2)

plot(score.norefit[8,], score.refit[8,], pch=c(15,17,19)[grp.prop.readm], bty="l", 
     xaxt="n", yaxt="n", ann=F,
     col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red")[grp.prop.readm])
abline(0,1)
axis(1, mgp=c(3,0.6,0))
axis(2, las=1, mgp=c(3,.7,0))
legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=cex, 
       pch=c(15,17,19), col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red"), 
       bty="n", title="readm rate")
mtext("model-based w/ model refitting", side=1, line=1.7)
mtext("model-based w/o model refitting", side=2, line=1.6)
dev.off()
# log link
system.time(res.Rcpp.log <- bin_log_cr(fail, Z, n_prov, time, 
                                       rep(-1, length(unique(data.prep[,fac.char]))), eta, beta))
# cloglog link
system.time(res.Rcpp.cloglog <-
              bin_cloglog_cr(fail, Z, n_prov, time, gamma, eta, beta))
# SRR with different links
pdf(paste0("SRR_CRM_difflinks", Sys.Date(), ".pdf"), height=9, width=9)
upper.panel <- function(x, y, ...) {
  points(x,y, pch=c(15,17,19)[grp.prop.readm], 
         col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red")[grp.prop.readm],
         xlim=c(-4,8), ylim=c(-4,8),...)
  abline(0, 1)
  r <- round(cor(x, y), digits=2)
  # txt <- paste0("CORR = ", r)
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # text(0.8, 0.1, txt, cex=1.5)
  legend("bottomright", legend=c("lowest 10%", "medium", "highest 10%"), cex=1.5,
         pch=c(15,17,19), col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red"), 
         bty="n", title="readm rate")
}

lower.panel <- function(x, y, ...) {
  points(x,y, pch=c(15,17,19)[grp.size], col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red")[grp.size],
         xlim=c(-4,8), ylim=c(-4,8),...)
  abline(0, 1)
  r <- round(cor(x, y), digits=2)
  # txt <- paste0("CORR = ", r)
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # text(0.8, 0.1, txt, cex=1.5)
  legend("topleft", legend=c("smallest 10%", "medium", "largest 10%"), cex=1.5,
         pch=c(15,17,19), col=c("blue", rgb(231,184,0,80,maxColorValue=255), "red"), 
         bty="n", title="discharge ct")
}

panel.hist <- function(x, ...) {
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  h <- hist(x, plot = F, breaks=100)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col="#00AFBB", ...)
}

data.for.pairs <- cbind(res.Rcpp.logit$SRR, res.Rcpp.log$SRR, res.Rcpp.cloglog$SRR)
colnames(data.for.pairs) <- c("SRR (logit)", "SRR (log)", "SRR (cloglog)")
pairs(data.for.pairs, lower.panel=lower.panel, 
      upper.panel=upper.panel, diag.panel=panel.hist, 
      gap=0.5, oma=c(2,2,2,2), cex.axis=cex)
dev.off()
# logistic regression SRR
res.bin <- logis.BIN.fe.prov(data.prep, Y.char, Z.char, fac.char, tol=1e-8, backtrack=T)
SRR.bin <- res.bin$df.prov$SRR

## diff in SRR vs. time to 1st event
pdf(paste0("SRRdiff_vs_timeto1stevent_Zscorediff_facsize", 
           Sys.Date(), ".pdf"), height=12, width=12)
par(mfrow=c(2,2))
idx <- order(abs(SRR.bin-res.Rcpp.logit$SRR), decreasing=T)
time.to.1stevent <- sapply(split(data.prep[,timetoevents.char], 
                                 data.prep[,fac.char]), mean)
diff.in.SRR <- res.Rcpp.logit$SRR-SRR.bin
pdf(paste0("SRRdiff_timeto1stevent", Sys.Date(), ".pdf"), height=4, width=6)
par(mar=c(2.5,3,0.3,0.3)+0.1)
plot(time.to.1stevent, diff.in.SRR, pch=c(15,17,19)[grp.prop.readm], bty="l", 
     xaxt="n", yaxt="n", ann=F, col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red")[grp.prop.readm])
axis(2, las=1, mgp=c(3,.7,0))
axis(1, mgp=c(3,0.6,0))
mtext("SRR (CRM) - SRR (LRM)", side=2, line=1.9)
mtext("average at-risk time (days)", side=1, line=1.5)
legend("bottomleft", legend=c("lowest 10%", "medium", "highest 10%"), 
       pch=c(15,17,19), col=c("blue", rgb(231,184,0,180,maxColorValue=255), "red"), 
       bty="n", title="readm rate")
dev.off()