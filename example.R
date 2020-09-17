m <- 1000 # number of providers
gamma1 <- -5.132825 # provider effect for provider 1
# n1 <- ls$n1 # size of provider 1
rho <- 0
n.rep <- 1000 # number of replicates
cutoff <- 10
pat.per.fac <- c(3, pmax(3, ceiling(rgamma(m-1, 2.588575, 0.06580117))))
fid <- paste0("f", rep(1:m, pat.per.fac))
pid <- paste0("p", do.call(c, lapply(pat.per.fac, function(i) 1:i)))
dis.per.pat <- sample.int(12, sum(pat.per.fac), T,
                          c(0.5387877036, 0.2348462947, 0.1138093861, 
                            0.0554110051, 0.0265009155, 0.0135877421, 
                            0.0074202563, 0.0037583117, 0.0024091741, 
                            0.0019273393, 0.0011564036, 0.0003854679))
invisible(sapply(names(which(sapply(split(dis.per.pat, fid),sum)<=cutoff)), 
                 function(name) {
                   tmp <- dis.per.pat[fid==name] 
                   tmp[1] <- tmp[1] + cutoff - sum(tmp) + 1
                   dis.per.pat[fid==name] <<- tmp
                 }))
fid <- rep(fid, dis.per.pat); pid <- rep(pid, dis.per.pat)
fpid <- paste0(fid, pid)
corr <- 0.5 # discharge-level correlation
mean.gamma <- -5.132825
sd.gamma <- 0.3062899 # based on real data
gamma <- c(gamma1, rnorm(m-1, mean.gamma, sd.gamma))
beta <- c(1,0.5,-1)
eta <- c(-0.02665738, -0.04123531, -0.06136877, -0.10437307, -0.17536328, 
         -0.11939965, -0.13998448, -0.15177138, -0.18690078, -0.22476105,
         -0.27071908, -0.29067531, -0.27614837, -0.30412037, -0.30398500,
         -0.30010643, -0.35833461, -0.37838084, -0.41938837, -0.38570910,
         -0.38652063, -0.42931252, -0.46088751, -0.45463939, -0.51339988, 
         -0.54059912)
readm.char <- 'readmitted'
prov.char <- 'prov.ID'
pat.char <- 'pat.ID'
time.char <- 'daysto1stevent'
Z.char <- paste0('z', 1:length(beta))
alpha <- 0.04

sim.comprisk <- function(m, fid, fpid, dis.per.pat, corr,
                         gamma, beta, eta, rho,
                         readm.char, Z.char, prov.char, time.char) {
  prov.size <- sapply(split(fid, fid), length)
  N <- sum(prov.size) # total number of discharges
  r <- length(beta)
  gamma.dis <- rep(gamma, times=prov.size)
  rZ <- function(i, rho, r)
    MASS::mvrnorm(n=prov.size[i],
                  mu=((gamma[i]-mean.gamma)*rho/sd.gamma)*matrix(1,nrow=r),
                  Sigma=diag(1-rho,r)+(rho-rho^2)*matrix(1,ncol=r,nrow=r))
  Z <- do.call(rbind, lapply(1:m, function(i) rZ(i,rho,r)))
  ranef <- do.call(c, lapply(dis.per.pat, 
                             function(d) {
                               Sigma <- matrix(corr, ncol=d, nrow=d)
                               diag(Sigma) <- 1
                               MASS::mvrnorm(1, rep(0, d), Sigma*0.09)
                             }))
  # time to first event
  rate.comp <- 0.05/(length(eta)+1) # observed rate from real data
  day <- 4
  idx.atrisk <- 1:N
  days.to.1stevent <- rep(day+length(eta), N)
  readm <- rep(0, N)
  idx.readm <- which(rbinom(N, 1, plogis(gamma.dis+Z%*%beta+ranef))==1)
  readm[idx.readm] <- 1
  days.to.1stevent[idx.readm] <- day
  idx.comp <- sample(idx.atrisk, round(rate.comp*length(idx.atrisk)))
  idx.out <- unique(c(idx.readm, idx.comp))
  for (x in eta) {
    day <- day + 1
    idx.atrisk <- c(1:N)[-idx.out]
    probs <- plogis(x+gamma.dis[idx.atrisk]+Z[idx.atrisk,]%*%beta+ranef[idx.atrisk])
    idx.readm <- idx.atrisk[rbinom(length(probs), 1, probs)==1]
    readm[idx.readm] <- 1
    days.to.1stevent[idx.readm] <- day
    idx.comp <- sample(idx.atrisk, round(rate.comp*length(idx.atrisk)))
    idx.out <- unique(c(idx.out, idx.readm, idx.comp))
  }
  data <- data.frame(readm, fid, fpid, Z, days.to.1stevent, stringsAsFactors=F)
  colnames(data) <- c(readm.char, prov.char, pat.char, Z.char, time.char)
  return(data)
} # end of sim.comprisk
data <- sim.comprisk(m, fid, fpid, dis.per.pat, corr,
                     gamma, beta, eta, rho,
                     readm.char, Z.char, prov.char, time.char)
# model fitting
Rcpp::sourceCpp('library.cpp')
prov.size <- sapply(split(data[,prov.char], data[,prov.char]), length)
cr.logit <- bin_logit_cr(data[,readm.char], as.matrix(data[,Z.char]), 
                         prov.size, data[,time.char], 
                         rep(0,length(gamma)), rep(0,length(eta)), 
                         rep(0,length(beta)))
cr.log <- bin_log_cr(data[,readm.char], as.matrix(data[,Z.char]), 
                     prov.size, data[,time.char], 
                     rep(-1,length(gamma)), rep(0,length(eta)), 
                     rep(0,length(beta)))
cr.cloglog <- bin_cloglog_cr(data[,readm.char], as.matrix(data[,Z.char]), 
                             prov.size, data[,time.char], 
                             rep(0,length(gamma)), rep(0,length(eta)), 
                             rep(0,length(beta)))
fail <- data[,readm.char]; Z <- as.matrix(data[,Z.char])
time <- data[,time.char]; prov <- data[,prov.char]
n_prov_pat <- lapply(split(data[, pat.char], data[, prov.char]), 
                     function(x) sapply(split(x, x), length))
# score tests
tests.score <- bin_logit_cr_score(fail, Z, prov.size, n_prov_pat, 
                                  time, cr.logit$gamma, cr.logit$eta, 
                                  cr.logit$beta, 1, alpha)