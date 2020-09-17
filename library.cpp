// #define ARMA_NO_DEBUG
// #define ARMA_DONT_USE_OPENMP
#define STRICT_R_HEADERS // needed on Windows, not on macOS
#include <RcppArmadillo.h>
#include <omp.h>
#include <cmath>
#include <vector>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// convert provider-specific gamma to discharge-specific gamma
vec gamma_provtodis(const vec &gamma, const IntegerVector &cumsum_prov) {
  int m = gamma.n_elem;
  vec gamma_dis(cumsum_prov[m]);
  for (int i = 0; i < m; i++) {
    gamma_dis.subvec(cumsum_prov[i], cumsum_prov[i+1]-1) = 
      gamma(i) * ones(cumsum_prov[i+1]-cumsum_prov[i]);
  }
  return gamma_dis;
}
// evaluate log-likelihood under logit link
double Loglkd_logit(const vec &fail, const mat &Z, 
                    const IntegerVector &cumsum_prov,
                    const vector<uvec> &idx_time,
                    const vec &gamma, const vec &eta, const vec &beta) {
  double obj = 0.0;
  // tmp1: gamma.dis + Z*beta
  vec tmp1 = gamma_provtodis(gamma, cumsum_prov) + Z*beta;
  vec tmp2(fail.n_elem); // tmp2: eta + gamma + Z*beta
  vec lincomb = tmp1;
  obj -= accu(log(exp(lincomb) + 1));
  tmp2.elem(idx_time[0]) = lincomb.elem(idx_time[0]);
  for (unsigned int i = 1; i < idx_time.size(); i++) {
    for (unsigned int k = 0; k < i; k++) {
      vec lincomb = eta(k) + tmp1.elem(idx_time[i]);
      if (k == i-1) tmp2.elem(idx_time[i]) = lincomb;
      obj -= accu(log(exp(lincomb) + 1));
    }
  }
  obj += accu(tmp2.elem(find(fail==1)));
  return obj;
}
// evaluate log-likelihood under log link
double Loglkd_log(const vec &fail, const mat &Z, 
                  const IntegerVector &cumsum_prov,
                  const vector<uvec> &idx_time,
                  const vec &gamma, const vec &eta, const vec &beta) {
  double obj = 0.0;
  // tmp1: gamma.dis + Z*beta
  vec tmp1 = gamma_provtodis(gamma, cumsum_prov) + Z*beta;
  // tmp2: eta + gamma + Z*beta, tmp3: log(1-exp(tmp2))
  vec tmp2(fail.n_elem), tmp3(fail.n_elem);
  vec lincomb = tmp1;
  vec logexplincomb = log(1-exp(lincomb));
  obj += accu(logexplincomb);
  tmp2.elem(idx_time[0]) = lincomb.elem(idx_time[0]);
  tmp3.elem(idx_time[0]) = logexplincomb.elem(idx_time[0]);
  for (unsigned int i = 1; i < idx_time.size(); i++) {
    for (unsigned int k = 0; k < i; k++) {
      vec lincomb = eta(k) + tmp1.elem(idx_time[i]);
      vec logexplincomb = log(1-exp(lincomb));
      if (k == i-1) {
        tmp2.elem(idx_time[i]) = lincomb;
        tmp3.elem(idx_time[i]) = logexplincomb;
      }
      obj += accu(logexplincomb);
    }
  }
  obj += accu(tmp2.elem(find(fail==1)));
  obj -= accu(tmp3.elem(find(fail==1)));
  return obj;
}

double Loglkd_cloglog(const vec &fail, const mat &Z, 
                      const IntegerVector &cumsum_prov,
                      const vector<uvec> &idx_time,
                      const vec &gamma, const vec &eta, const vec &beta) {
  double obj = 0.0;
  // tmp1: gamma.dis + Z*beta
  vec tmp1 = gamma_provtodis(gamma, cumsum_prov) + Z*beta;
  // tmp2: exp(eta + gamma + Z*beta), tmp3: log(1-exp(-exp(tmp2)))
  vec tmp2(fail.n_elem), tmp3(fail.n_elem);
  vec explincomb = exp(tmp1);
  vec logexpexplincomb = log(1-exp(-explincomb));
  obj -= accu(explincomb);
  tmp2.elem(idx_time[0]) = explincomb.elem(idx_time[0]);
  tmp3.elem(idx_time[0]) = logexpexplincomb.elem(idx_time[0]);
  for (unsigned int i = 1; i < idx_time.size(); i++) {
    for (unsigned int k = 0; k < i; k++) {
      vec explincomb = exp(eta(k) + tmp1.elem(idx_time[i]));
      vec logexpexplincomb = log(1-exp(-explincomb));
      if (k == i-1) {
        tmp2.elem(idx_time[i]) = explincomb;
        tmp3.elem(idx_time[i]) = logexpexplincomb;
      }
      obj -= accu(explincomb);
    }
  }
  obj += accu(tmp2.elem(find(fail==1)));
  obj += accu(tmp3.elem(find(fail==1)));
  return obj;
}

double Loglkd_probit(const vec &fail, const mat &Z, 
                     const IntegerVector &cumsum_prov,
                     const vector<uvec> &idx_time,
                     const vec &gamma, const vec &eta, const vec &beta) {
  double obj = 0.0;
  // tmp1: gamma.dis + Z*beta
  vec tmp1 = gamma_provtodis(gamma, cumsum_prov) + Z*beta;
  // tmp2: log(1/(1-Phi(eta + gamma + Z*beta))-1)
  vec tmp2(fail.n_elem);
  vec oneminusphilincomb = 1-normcdf(tmp1);
  obj += accu(log(oneminusphilincomb));
  tmp2.elem(idx_time[0]) = log(1/oneminusphilincomb.elem(idx_time[0])-1);
  
  for (unsigned int i = 1; i < idx_time.size(); i++) {
    for (unsigned int k = 0; k < i; k++) {
      vec oneminusphilincomb = 1-normcdf(eta(k) + tmp1.elem(idx_time[i]));
      if (k == i-1) {
        tmp2.elem(idx_time[i]) = log(1/oneminusphilincomb-1);
      }
      obj += accu(log(oneminusphilincomb));
    }
  }
  obj += accu(tmp2.elem(find(fail==1)));
  return obj;
}

double ifelse_scalar(bool test, double yes, double no) {
  if (test) return(yes);
  else return(no);
}

// comparable to the R version
NumericVector unlist(const List& list) { // from Kevin Ushey
  std::size_t n = list.size();
  // Figure out the length of the output vector
  std::size_t total_length = 0;
  for (std::size_t i = 0; i < n; ++i)
    total_length += Rf_length(list[i]);
  // Allocate the vector
  NumericVector output = no_init(total_length);
  // Loop and fill
  std::size_t index = 0;
  for (std::size_t i = 0; i < n; ++i)
  {
    NumericVector el = list[i];
    std::copy(el.begin(), el.end(), output.begin() + index);
    
    // Update the index
    index += el.size();
  }
  return output;
}

// [[Rcpp::export]]
List bin_logit_cr(vec &fail, mat &Z, IntegerVector &n_prov, vec &time,
                  vec gamma, vec eta, vec beta, 
                  int parallel=1, int threads=1,
                  double s=0.01, double t=0.6, 
                  double tol=1e-8, int max_iter=100, double bound=10.0,
                  double alpha=0.05) {
  
  unsigned int m = gamma.n_elem, n = Z.n_rows; // prov/sample count
  unsigned int p = beta.n_elem, tau = eta.n_elem; // covar/time count
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vec uniqtime = unique(time), obs_time(tau+1); // count by time
  vector<uvec> idx_time; // index of disc stratified by time
  for (unsigned int i = 0; i <= tau; i++) {
    uvec idx_tmp = find(time == uniqtime(i));
    idx_time.push_back(idx_tmp);
    obs_time(i) = accu(fail.elem(idx_tmp));
  }
  vec obs_prov(m); // failure count by provider
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
  }
  double d_obj, v, inc, crit = 100.0;
  double obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                            gamma, eta, beta);// initialize objetive function
  int iter = 0, btr_max = 1000, btr_ct = 0;
  cout << "Implementing BIN (logit) ..." << endl;
  while (iter < max_iter && crit >= tol) {
    iter++;
    // calculating scores and info matrices
    vec tmp = gamma_provtodis(gamma, cumsum_prov) + Z * beta;
    // sum of p_{ij}(k) or q_{ij}(k) over k for each dis
    vec psum = 1 / (1 + exp(-tmp)); // initialized at time k = 1
    vec qsum = psum % (1 - psum); // initialized at time k = 1
    vec score_eta(tau, fill::zeros), info_eta(tau, fill::zeros);
    mat info_beta_eta(p, tau, fill::zeros);
    mat tmpmat(n, tau, fill::zeros); // tmpmat to info_eta_gamma
    for (unsigned int i = 1; i <= tau; i++) {
      score_eta(i-1) += obs_time(i);
      mat Z_tmp = Z.rows(idx_time[i]); 
      for (unsigned int k = 0; k < i; k++) { // at risk
        vec p_tmp = 1 / (1+exp(-eta(k)-tmp.elem(idx_time[i])));
        vec q_tmp = p_tmp % (1 - p_tmp);
        psum.elem(idx_time[i]) += p_tmp;
        qsum.elem(idx_time[i]) += q_tmp;
        score_eta(k) -= accu(p_tmp);
        info_eta(k) += accu(q_tmp);
        uvec col_tmpmat(1); col_tmpmat.fill(k);
        tmpmat.submat(idx_time[i], col_tmpmat) += q_tmp;
        info_beta_eta.col(k) += sum(Z_tmp.each_col() % q_tmp).t();
        // note: a non-contiguous submatrix doesn't have each_col as 
        //       its member function
      }
    }
    mat Z_sum = Z.each_col() % psum; // Z % psum
    vec score_beta = sum(Z.rows(find(fail==1))).t() - sum(Z_sum).t();
    Z_sum = Z.each_col() % qsum; // Z % qsum
    mat info_beta = Z.t() * Z_sum;
    vec score_gamma(m), info_gamma_inv(m);
    mat info_eta_gamma(tau, m), info_beta_gamma(p, m);
    for (unsigned int i = 0; i < m; i++) {
      unsigned int start = cumsum_prov[i]; 
      unsigned int   end = cumsum_prov[i+1]-1;
      score_gamma(i) = obs_prov(i) - accu(psum.subvec(start, end));
      info_gamma_inv(i) = 1 / accu(qsum.subvec(start, end));
      info_eta_gamma.col(i) = sum(tmpmat.rows(start, end)).t();
      info_beta_gamma.col(i) = sum(Z_sum.rows(start, end)).t();
    }
    // Newton step
    mat info_1 = join_cols(info_eta_gamma.each_row()%info_gamma_inv.t(),
                           info_beta_gamma.each_row()%info_gamma_inv.t());
    mat schur_inv = inv_sympd(join_rows(
      join_cols(diagmat(info_eta),info_beta_eta),
      join_cols(info_beta_eta.t(),info_beta)) - 
        join_cols(info_eta_gamma,info_beta_gamma)*info_1.t());
    mat info_2 = schur_inv * info_1;
    vec d_gamma = info_gamma_inv % score_gamma + 
      info_2.t()*(info_1*score_gamma-join_cols(score_eta,score_beta));
    vec d_etabeta = schur_inv*join_cols(score_eta,score_beta) - 
      info_2*score_gamma;
    vec d_eta = d_etabeta.head(tau), d_beta = d_etabeta.tail(p);
    // backtracking line search
    v = 1.0; // initial step size
    d_obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                         gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    inc = dot(score_gamma, d_gamma) + dot(score_eta, d_eta) +
      dot(score_beta, d_beta); // square of Newton increment
    int btr_ct_iter = 0, btr_max_iter = 50;
    while (d_obj < s*v*inc && btr_ct < btr_max && btr_ct_iter < btr_max_iter) {
      ++btr_ct;
      ++btr_ct_iter;
      v *= t;
      d_obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                           gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    }
    gamma += v * d_gamma;
    double med_gamma = median(gamma);
    gamma = clamp(gamma, med_gamma-bound, med_gamma+bound);
    beta += v * d_beta;
    eta += v * d_eta;
    obj += d_obj;
    
    crit = v * norm(d_etabeta, "inf");
    cout << "Iter " << iter << ": loglkd = " << scientific
         << setprecision(5) << obj << ";";
    cout << " running diff = " << scientific
         << setprecision(5) << crit << ";" << endl;
  }
  cout << "BIN converged after " << iter << " iterations!" << endl;
  // SRR and indep score test
  vec tmp = gamma_provtodis(median(gamma)*ones(m), cumsum_prov) + Z*beta;
  vec tmp_pred = gamma_provtodis(gamma, cumsum_prov) + Z*beta;
  vec hazsum = 1 / (1 + exp(-tmp)); // initialized at time k = 1
  vec hazsum_pred = 1 / (1 + exp(-tmp_pred));
  vec qsum = hazsum % (1 - hazsum); // initialized at time k = 1
  vec qsum_pred = hazsum_pred % (1 - hazsum_pred);
  // mat exact_tmpmat(n, tau+1, fill::zeros); // exact test
  // exact_tmpmat.col(0) += hazsum;
  for (unsigned int i = 1; i <= tau; i++) {
    for (unsigned int k = 0; k < i; k++) { // at risk
      vec haz_tmp = 1 / (1+exp(-eta(k)-tmp.elem(idx_time[i])));
      vec haz_tmp_pred = 1 / (1+exp(-eta(k)-tmp_pred.elem(idx_time[i])));
      hazsum.elem(idx_time[i]) += haz_tmp;
      hazsum_pred.elem(idx_time[i]) += haz_tmp_pred;
      qsum.elem(idx_time[i]) += haz_tmp % (1 - haz_tmp);
      qsum_pred.elem(idx_time[i]) += haz_tmp_pred % (1 - haz_tmp_pred);
      // uvec col_tmpmat(1); col_tmpmat.fill(k+1);
      // exact_tmpmat.submat(idx_time[i], col_tmpmat) += haz_tmp;
    }
  }
  vec exp_prov(m), pred_prov(m); // expected readmin
  vec scorestat_prov(m), scoreflag_prov(m), scorepval_prov(m); // score
  // vec exactpval_prov(m), exactflag_prov(m); // exact test p-values/flags
  // // load package poibin
  // Environment poibin = Environment::namespace_env("poibin");
  // Function ppoibin = poibin["ppoibin"], dpoibin = poibin["dpoibin"];
  
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    // exp readmin
    exp_prov(i) = accu(hazsum.subvec(start, end));
    // pred readmin
    pred_prov(i) = accu(hazsum_pred.subvec(start, end));
    // score test
    scorestat_prov(i) = (obs_prov(i) - exp_prov(i))
      / sqrt(accu(qsum.subvec(start, end)));
    double prob = 1-normcdf(scorestat_prov(i)); // score test upper tail
    scoreflag_prov(i) =
      ifelse_scalar(prob<alpha/2, 1, ifelse_scalar(prob<=1-alpha/2, 0, -1));
    scorepval_prov(i) = 2*min(prob, 1-prob);
    // // exact test
    // vec probs = nonzeros(exact_tmpmat.rows(start, end));
    // prob =  // exact test upper tail
    //   1 - as<double>(ppoibin(obs_prov(i), probs, "DFT-CF", R_NilValue))
    //   + 0.5*as<double>(dpoibin(obs_prov(i), probs, R_NilValue));
    // exactflag_prov(i) =
    //   ifelse_scalar(prob<alpha/2, 1, ifelse_scalar(prob<=1-alpha/2, 0, -1));
    // exactpval_prov(i) = 2*min(prob, 1-prob);  
  }
  
  List ret = List::create(_["gamma"]=gamma, 
                          _["eta"]=eta, _["beta"]=beta,
                          _["Obs"]=obs_prov, _["Exp"]=exp_prov,
                          _["Pred"]=pred_prov,
                          _["SRR"]=obs_prov/exp_prov,
                          _["score.stat"]=scorestat_prov,
                          _["score.flag"]=scoreflag_prov,
                          _["score.pval"]=scorepval_prov);
  // _["exact.flag"]=exactflag_prov,
  // _["exact.pval"]=exactpval_prov
  return ret;
}

// [[Rcpp::export]]
List bin_logit_cr_zeroeta(vec &fail, mat &Z, IntegerVector &n_prov, vec &time,
                          vec gamma, vec eta, vec beta, 
                          int parallel=1, int threads=1,
                          double s=0.01, double t=0.6, 
                          double tol=1e-8, int max_iter=100, double bound=10.0,
                          double alpha=0.05) {
  
  unsigned int m = gamma.n_elem, n = Z.n_rows; // prov/sample count
  unsigned int p = beta.n_elem, tau = eta.n_elem; // covar/time count
  eta.zeros();
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vec uniqtime = unique(time), obs_time(tau+1); // count by time
  vector<uvec> idx_time; // index of disc stratified by time
  for (unsigned int i = 0; i <= tau; i++) {
    uvec idx_tmp = find(time == uniqtime(i));
    idx_time.push_back(idx_tmp);
    obs_time(i) = accu(fail.elem(idx_tmp));
  }
  vec obs_prov(m); // failure count by provider
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
  }
  double d_obj, v, inc, crit = 100.0;
  double obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                            gamma, eta, beta);// initialize objetive function
  int iter = 0, btr_max = 1000, btr_ct = 0;
  cout << "Implementing BIN (logit) ..." << endl;
  while (iter < max_iter && crit >= tol) {
    iter++;
    // calculating scores and info matrices
    vec tmp = gamma_provtodis(gamma, cumsum_prov) + Z * beta;
    // sum of p_{ij}(k) or q_{ij}(k) over k for each dis
    vec psum = 1 / (1 + exp(-tmp)); // initialized at time k = 1
    vec qsum = psum % (1 - psum); // initialized at time k = 1
    vec score_eta(tau, fill::zeros), info_eta(tau, fill::zeros);
    mat info_beta_eta(p, tau, fill::zeros);
    mat tmpmat(n, tau, fill::zeros); // tmpmat to info_eta_gamma
    for (unsigned int i = 1; i <= tau; i++) {
      score_eta(i-1) += obs_time(i);
      mat Z_tmp = Z.rows(idx_time[i]); 
      for (unsigned int k = 0; k < i; k++) { // at risk
        vec p_tmp = 1 / (1+exp(-eta(k)-tmp.elem(idx_time[i])));
        vec q_tmp = p_tmp % (1 - p_tmp);
        psum.elem(idx_time[i]) += p_tmp;
        qsum.elem(idx_time[i]) += q_tmp;
        score_eta(k) -= accu(p_tmp);
        info_eta(k) += accu(q_tmp);
        uvec col_tmpmat(1); col_tmpmat.fill(k);
        tmpmat.submat(idx_time[i], col_tmpmat) += q_tmp;
        info_beta_eta.col(k) += sum(Z_tmp.each_col() % q_tmp).t();
        // note: a non-contiguous submatrix doesn't have each_col as 
        //       its member function
      }
    }
    mat Z_sum = Z.each_col() % psum; // Z % psum
    vec score_beta = sum(Z.rows(find(fail==1))).t() - sum(Z_sum).t();
    Z_sum = Z.each_col() % qsum; // Z % qsum
    mat info_beta = Z.t() * Z_sum;
    vec score_gamma(m), info_gamma_inv(m);
    mat info_eta_gamma(tau, m), info_beta_gamma(p, m);
    for (unsigned int i = 0; i < m; i++) {
      unsigned int start = cumsum_prov[i]; 
      unsigned int   end = cumsum_prov[i+1]-1;
      score_gamma(i) = obs_prov(i) - accu(psum.subvec(start, end));
      info_gamma_inv(i) = 1 / accu(qsum.subvec(start, end));
      info_eta_gamma.col(i) = sum(tmpmat.rows(start, end)).t();
      info_beta_gamma.col(i) = sum(Z_sum.rows(start, end)).t();
    }
    info_eta_gamma.zeros(); info_eta.zeros(); info_beta_eta.zeros();
    score_eta.zeros();
    // Newton step
    mat info_1 = info_beta_gamma.each_row()%info_gamma_inv.t();
    mat schur_inv = inv_sympd(info_beta - info_beta_gamma*info_1.t());
    mat info_2 = schur_inv * info_1;
    vec d_gamma = info_gamma_inv % score_gamma + 
      info_2.t()*(info_1*score_gamma-score_beta);
    vec d_beta = schur_inv*score_beta - info_2*score_gamma;
    // backtracking line search
    v = 1.0; // initial step size
    d_obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                         gamma+v*d_gamma, eta, beta+v*d_beta) - obj;
    inc = dot(score_gamma, d_gamma) + dot(score_beta, d_beta); // square of Newton increment
    int btr_ct_iter = 0, btr_max_iter = 50;
    while (d_obj < s*v*inc && btr_ct < btr_max && btr_ct_iter < btr_max_iter) {
      ++btr_ct;
      ++btr_ct_iter;
      v *= t;
      d_obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                           gamma+v*d_gamma, eta, beta+v*d_beta) - obj;
    }
    gamma += v * d_gamma;
    double med_gamma = median(gamma);
    gamma = clamp(gamma, med_gamma-bound, med_gamma+bound);
    beta += v * d_beta;
    obj += d_obj;
    
    crit = v * norm(d_beta, "inf");
    cout << "Iter " << iter << ": loglkd = " << scientific
         << setprecision(5) << obj << ";";
    cout << " running diff = " << scientific
         << setprecision(5) << crit << ";" << endl;
  }
  cout << "BIN converged after " << iter << " iterations!" << endl;
  // SRR and indep score test
  vec tmp = gamma_provtodis(median(gamma)*ones(m), cumsum_prov) + Z*beta;
  vec tmp_pred = gamma_provtodis(gamma, cumsum_prov) + Z*beta;
  vec hazsum = 1 / (1 + exp(-tmp)); // initialized at time k = 1
  vec hazsum_pred = 1 / (1 + exp(-tmp_pred));
  vec qsum = hazsum % (1 - hazsum); // initialized at time k = 1
  vec qsum_pred = hazsum_pred % (1 - hazsum_pred);
  // mat exact_tmpmat(n, tau+1, fill::zeros); // exact test
  // exact_tmpmat.col(0) += hazsum;
  for (unsigned int i = 1; i <= tau; i++) {
    for (unsigned int k = 0; k < i; k++) { // at risk
      vec haz_tmp = 1 / (1+exp(-eta(k)-tmp.elem(idx_time[i])));
      vec haz_tmp_pred = 1 / (1+exp(-eta(k)-tmp_pred.elem(idx_time[i])));
      hazsum.elem(idx_time[i]) += haz_tmp;
      hazsum_pred.elem(idx_time[i]) += haz_tmp_pred;
      qsum.elem(idx_time[i]) += haz_tmp % (1 - haz_tmp);
      qsum_pred.elem(idx_time[i]) += haz_tmp_pred % (1 - haz_tmp_pred);
      // uvec col_tmpmat(1); col_tmpmat.fill(k+1);
      // exact_tmpmat.submat(idx_time[i], col_tmpmat) += haz_tmp;
    }
  }
  vec exp_prov(m), pred_prov(m); // expected readmin
  vec scorestat_prov(m), scoreflag_prov(m), scorepval_prov(m); // score
  // vec exactpval_prov(m), exactflag_prov(m); // exact test p-values/flags
  // // load package poibin
  // Environment poibin = Environment::namespace_env("poibin");
  // Function ppoibin = poibin["ppoibin"], dpoibin = poibin["dpoibin"];
  
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    // exp readmin
    exp_prov(i) = accu(hazsum.subvec(start, end));
    // pred readmin
    pred_prov(i) = accu(hazsum_pred.subvec(start, end));
    // score test
    scorestat_prov(i) = (obs_prov(i) - exp_prov(i))
      / sqrt(accu(qsum.subvec(start, end)));
    double prob = 1-normcdf(scorestat_prov(i)); // score test upper tail
    scoreflag_prov(i) =
      ifelse_scalar(prob<alpha/2, 1, ifelse_scalar(prob<=1-alpha/2, 0, -1));
    scorepval_prov(i) = 2*min(prob, 1-prob);
    // // exact test
    // vec probs = nonzeros(exact_tmpmat.rows(start, end));
    // prob =  // exact test upper tail
    //   1 - as<double>(ppoibin(obs_prov(i), probs, "DFT-CF", R_NilValue))
    //   + 0.5*as<double>(dpoibin(obs_prov(i), probs, R_NilValue));
    // exactflag_prov(i) =
    //   ifelse_scalar(prob<alpha/2, 1, ifelse_scalar(prob<=1-alpha/2, 0, -1));
    // exactpval_prov(i) = 2*min(prob, 1-prob);  
  }
  
  List ret = List::create(_["gamma"]=gamma, 
                          _["eta"]=eta, _["beta"]=beta,
                          _["Obs"]=obs_prov, _["Exp"]=exp_prov,
                          _["Pred"]=pred_prov,
                          _["SRR"]=obs_prov/exp_prov,
                          _["score.stat"]=scorestat_prov,
                          _["score.flag"]=scoreflag_prov,
                          _["score.pval"]=scorepval_prov);
  // _["exact.flag"]=exactflag_prov,
  // _["exact.pval"]=exactpval_prov
  return ret;
}

// [[Rcpp::export]]
List bin_logit_cr_cstr(vec &fail, mat &Z, IntegerVector &n_prov, vec &time, 
                       vec gamma, vec eta, vec beta, 
                       unsigned int idx_gamma, double cstr_gamma,
                       double s=0.01, double t=0.6, 
                       double tol=1e-8, int max_iter=100, 
                       double bound=10.0, double alpha=0.05) {
  
  unsigned int m = gamma.n_elem, n = Z.n_rows; // prov/sample count
  if (idx_gamma < 1 || idx_gamma > m) { // R's one-based indexing
    cout << "Error: Argument 'idx_gamma' out of range!" << endl;
    return -1;
  }
  idx_gamma -= 1;
  gamma(idx_gamma) = cstr_gamma; // replace that gamma with constraint
  unsigned int p = beta.n_elem, tau = eta.n_elem; // covar/time count
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vec uniqtime = unique(time), obs_time(tau+1); // count by time
  vector<uvec> idx_time; // index of disc stratified by time
  for (unsigned int i = 0; i <= tau; i++) {
    uvec idx_tmp = find(time == uniqtime(i));
    idx_time.push_back(idx_tmp);
    obs_time(i) = accu(fail.elem(idx_tmp));
  }
  vec obs_prov(m); // failure count by provider
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
  }
  double d_obj, v, inc, crit = 100.0;
  double obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                            gamma, eta, beta);// initialize objetive function
  int iter = 0;
  cout << "Implementing BIN (logit) ..." << endl;
  while (iter < max_iter && crit >= tol) {
    iter++;
    // calculating scores and info matrices
    vec tmp = gamma_provtodis(gamma, cumsum_prov) + Z * beta;
    // sum of p_{ij}(k) or q_{ij}(k) over k for each dis
    vec psum = 1 / (1 + exp(-tmp)); // initialized at time k = 1
    vec qsum = psum % (1 - psum); // initialized at time k = 1
    vec score_eta(tau, fill::zeros), info_eta(tau, fill::zeros);
    mat info_beta_eta(p, tau, fill::zeros);
    mat tmpmat(n, tau, fill::zeros); // tmpmat to info_eta_gamma
    for (unsigned int i = 1; i <= tau; i++) {
      score_eta(i-1) += obs_time(i);
      mat Z_tmp = Z.rows(idx_time[i]); 
      for (unsigned int k = 0; k < i; k++) { // at risk
        vec p_tmp = 1 / (1+exp(-eta(k)-tmp.elem(idx_time[i])));
        vec q_tmp = p_tmp % (1 - p_tmp);
        psum.elem(idx_time[i]) += p_tmp;
        qsum.elem(idx_time[i]) += q_tmp;
        score_eta(k) -= accu(p_tmp);
        info_eta(k) += accu(q_tmp);
        uvec col_tmpmat(1); col_tmpmat.fill(k);
        tmpmat.submat(idx_time[i], col_tmpmat) += q_tmp;
        info_beta_eta.col(k) += sum(Z_tmp.each_col() % q_tmp).t();
        // note: a non-contiguous submatrix doesn't have each_col as 
        //       its member function
      }
    }
    mat Z_sum = Z.each_col() % psum; // Z % psum
    vec score_beta = sum(Z.rows(find(fail==1))).t() - sum(Z_sum).t();
    Z_sum = Z.each_col() % qsum; // Z % qsum
    mat info_beta = Z.t() * Z_sum;
    vec score_gamma(m-1), info_gamma_inv(m-1);
    mat info_eta_gamma(tau, m-1), info_beta_gamma(p, m-1);
    for (unsigned int i = 0; i < idx_gamma; i++) {
      unsigned int start = cumsum_prov[i]; 
      unsigned int   end = cumsum_prov[i+1]-1;
      score_gamma(i) = obs_prov(i) - accu(psum.subvec(start, end));
      info_gamma_inv(i) = 1 / accu(qsum.subvec(start, end));
      info_eta_gamma.col(i) = sum(tmpmat.rows(start, end)).t();
      info_beta_gamma.col(i) = sum(Z_sum.rows(start, end)).t();
    }
    for (unsigned int i = idx_gamma+1; i < m; i++) {
      unsigned int start = cumsum_prov[i]; 
      unsigned int   end = cumsum_prov[i+1]-1;
      score_gamma(i-1) = obs_prov(i) - accu(psum.subvec(start, end));
      info_gamma_inv(i-1) = 1 / accu(qsum.subvec(start, end));
      info_eta_gamma.col(i-1) = sum(tmpmat.rows(start, end)).t();
      info_beta_gamma.col(i-1) = sum(Z_sum.rows(start, end)).t();
    }
    // Newton step
    mat info_1 = join_cols(info_eta_gamma.each_row()%info_gamma_inv.t(),
                           info_beta_gamma.each_row()%info_gamma_inv.t());
    mat schur_inv = inv_sympd(join_rows(
      join_cols(diagmat(info_eta),info_beta_eta),
      join_cols(info_beta_eta.t(),info_beta)) - 
        join_cols(info_eta_gamma,info_beta_gamma)*info_1.t());
    mat info_2 = schur_inv * info_1;
    vec d_gamma = info_gamma_inv % score_gamma + 
      info_2.t()*(info_1*score_gamma-join_cols(score_eta,score_beta));
    vec d_gamma_tmp = d_gamma; // one less than d_gamma
    d_gamma.insert_rows(idx_gamma, 1); // insert zero at idx_gamma
    vec d_etabeta = schur_inv*join_cols(score_eta,score_beta) - 
      info_2*score_gamma;
    vec d_eta = d_etabeta.head(tau), d_beta = d_etabeta.tail(p);
    // backtracking line search
    v = 1.0; // initial step size
    d_obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                         gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    inc = dot(score_gamma, d_gamma_tmp) + dot(score_eta, d_eta) +
      dot(score_beta, d_beta); // square of Newton increment
    while (d_obj < s*v*inc) {
      v *= t;
      d_obj = Loglkd_logit(fail, Z, cumsum_prov, idx_time, 
                           gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    }
    gamma += v * d_gamma;
    double med_gamma = median(gamma);
    gamma = clamp(gamma, med_gamma-bound, med_gamma+bound);
    beta += v * d_beta;
    eta += v * d_eta;
    obj += d_obj;
    
    crit = v * norm(d_etabeta, "inf");
    cout << "Iter " << iter << ": loglkd = " << scientific
         << setprecision(5) << obj << ";";
    cout << " running diff = " << scientific
         << setprecision(5) << crit << ";" << endl;
  }
  cout << "BIN converged after " << iter << " iterations!" << endl;
  // SRR and indep score test
  // vec tmp = gamma_provtodis(median(gamma)*ones(m), cumsum_prov) + Z*beta;
  // vec hazsum = 1 / (1 + exp(-tmp)); // initialized at time k = 1
  // vec qsum = hazsum % (1 - hazsum); // initialized at time k = 1
  // mat exact_tmpmat(n, tau+1, fill::zeros); // exact test
  // exact_tmpmat.col(0) += hazsum;
  // for (unsigned int i = 1; i <= tau; i++) {
  //   for (unsigned int k = 0; k < i; k++) { // at risk
  //     vec haz_tmp = 1 / (1+exp(-eta(k)-tmp.elem(idx_time[i])));
  //     hazsum.elem(idx_time[i]) += haz_tmp;
  //     qsum.elem(idx_time[i]) += haz_tmp % (1 - haz_tmp);
  //     uvec col_tmpmat(1); col_tmpmat.fill(k+1);
  //     exact_tmpmat.submat(idx_time[i], col_tmpmat) += haz_tmp;
  //   }
  // }
  // vec exp_prov(m); // expected readmin
  // vec scorestat_prov(m), scoreflag_prov(m), scorepval_prov(m); // score
  // 
  // for (unsigned int i = 0; i < m; i++) {
  //   unsigned int start = cumsum_prov[i]; 
  //   unsigned int   end = cumsum_prov[i+1]-1;
  //   // exp readmin
  //   exp_prov(i) = accu(hazsum.subvec(start, end));
  //   // score test
  //   scorestat_prov(i) = (obs_prov(i) - exp_prov(i))
  //     / sqrt(accu(qsum.subvec(start, end)));
  //   double prob = 1-normcdf(scorestat_prov(i)); // score test upper tail
  //   scoreflag_prov(i) =
  //     ifelse_scalar(prob<alpha/2, 1, ifelse_scalar(prob<=1-alpha/2, 0, -1));
  //   scorepval_prov(i) = 2*min(prob, 1-prob);
  // }
  List ret = List::create(_["gamma"]=gamma, _["eta"]=eta, _["beta"]=beta);
  // _["Obs"]=obs_prov, _["Exp"]=exp_prov,
  // _["SRR"]=obs_prov/exp_prov,
  // _["score.stat"]=scorestat_prov,
  // _["score.flag"]=scoreflag_prov,
  // _["score.pval"]=scorepval_prov
  
  return ret;
}

// [[Rcpp::export]]
NumericVector bin_logit_cr_score(vec &fail, mat &Z, IntegerVector &n_prov, 
                                 List n_prov_pat, vec &time, vec gamma, vec eta, 
                                 vec beta, unsigned int idx_gamma, 
                                 double alpha=0.05) {
  idx_gamma -= 1;
  gamma(idx_gamma) = median(gamma);
  unsigned int m = gamma.n_elem, n = Z.n_rows; // prov/sample count
  unsigned int p = beta.n_elem, tau = eta.n_elem; // covar/time count
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vec uniqtime = unique(time), obs_time(tau+1); // count by time
  vector<uvec> idx_time; // index of disc stratified by time
  int denom_phi = -(m + tau + p); // denominator of phi
  for (unsigned int i = 0; i <= tau; i++) {
    uvec idx_tmp = find(time == uniqtime(i));
    idx_time.push_back(idx_tmp);
    obs_time(i) = accu(fail.elem(idx_tmp));
    denom_phi += (i + 1) * idx_tmp.n_elem;
  }
  vec obs_prov(m); // failure count by provider
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
  }
  vec tmp = gamma_provtodis(gamma, cumsum_prov) + Z * beta;
  vec oddsum = exp(tmp); // h/(1-h) for squared Pearson resid
  vec oddsum_t0 = oddsum.elem(idx_time[0]);
  vec oddsum_t0_fail = oddsum_t0.elem(find(fail.elem(idx_time[0])==1));
  double phi = accu(1/oddsum_t0_fail - oddsum_t0_fail); // sum of (1-2h)/[h(1-h)]
  // sum of u_{ij}(k) or h'_{ij}(k) over k for each dis
  vec usum = 1 / (1 + 1/oddsum); // for meat, init at time k = 1
  vec vsum = usum % (1 - usum); // for bread, init at time k = 1
  // vec tmp_phi = -sqrt(oddsum);
  // tmp_phi.elem(idx_time[0]) += 
  //   fail.elem(idx_time[0]) / sqrt(vsum.elem(idx_time[0]));
  vec bread_eta(tau, fill::zeros);
  mat bread_beta_eta(p, tau, fill::zeros);
  mat tmpcov(n, tau+1, fill::zeros); // for cov matrix, see Pan(2001b)
  tmpcov.col(0) -= sqrt(oddsum);
  uvec col_cov(1); col_cov.fill(0);
  tmpcov.submat(idx_time[0], col_cov) +=
    fail.elem(idx_time[0]) / sqrt(vsum.elem(idx_time[0]));
  mat tmpmeat(n, tau, fill::zeros); // for meat
  mat panmeat(n, tau+1, fill::zeros); // for meat by Pan (2001b)
  panmeat.col(0) = sqrt(vsum);
  mat tmp_pm(n, tau, fill::zeros); // for meat by Pan (2001b)
  mat tmpmat(n, tau, fill::zeros); // for bread
  for (unsigned int i = 1; i <= tau; i++) {
    mat Z_tmp = Z.rows(idx_time[i]); // for bread
    uvec col_meat(1); col_meat.fill(i-1);
    tmpmeat.submat(idx_time[i], col_meat) += fail.elem(idx_time[i]);
    for (unsigned int k = 0; k < tau; k++) {
      vec odds_tmp = exp(eta(k)+tmp.elem(idx_time[i]));
      vec h_tmp = 1 / (1 + 1/odds_tmp);
      vec v_tmp = h_tmp % (1 - h_tmp);
      uvec col(1); col.fill(k);
      // for panmeat
      panmeat.submat(idx_time[i], col+1) += sqrt(v_tmp);
      if (k < i) { // at risk
        // meat
        usum.elem(idx_time[i]) += h_tmp;
        tmpmeat.submat(idx_time[i], col) -= h_tmp;
        // bread
        vsum.elem(idx_time[i]) += v_tmp;
        bread_eta(k) += accu(v_tmp);
        tmpmat.submat(idx_time[i], col) += v_tmp;
        bread_beta_eta.col(k) += sum(Z_tmp.each_col() % v_tmp).t();
        // phi
        oddsum.elem(idx_time[i]) += odds_tmp;
        // tmp_phi.elem(idx_time[i]) -= sqrt(odds_tmp);
        // cov
        tmpcov.submat(idx_time[i], col+1) -= sqrt(odds_tmp);
        if (k==i-1) {
          // tmp_phi.elem(idx_time[i]) += fail.elem(idx_time[i])/sqrt(v_tmp);
          vec odds_temp = odds_tmp.elem(find(fail.elem(idx_time[i])==1));
          phi += accu(1/odds_temp - odds_temp);
          // cov
          tmpcov.submat(idx_time[i], col+1) += 
            fail.elem(idx_time[i]) / sqrt(v_tmp);
          // panmeat
          tmp_pm.submat(idx_time[i], col) += sqrt(v_tmp);
        }        
      }
    }
  }
  // phi
  phi += accu(oddsum);
  phi /= denom_phi;
  // cov
  unsigned int dim_cov = max(unlist(n_prov_pat))*(tau+1);
  // stratified cov
  IntegerVector disc_ct = table(unlist(n_prov_pat));
  vector<mat> set_cov;
  for (unsigned int i = 0; i < disc_ct.size(); i++) {
    mat cov_mat_tmp((i+1)*(tau+1), (i+1)*(tau+1), fill::zeros);
    set_cov.push_back(cov_mat_tmp);
  }
  mat cov(dim_cov, dim_cov, fill::zeros);
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start_prov = cumsum_prov[i];
    unsigned int   end_prov = cumsum_prov[i+1]-1;
    mat tmpcov_tmp = tmpcov.rows(start_prov, end_prov);
    IntegerVector cumsum_tmp = cumsum(as<IntegerVector>(n_prov_pat[i]));
    unsigned int pats = cumsum_tmp.size(); // patient ct
    
    cumsum_tmp.push_front(0);
    for (unsigned int j = 0; j < pats; j++) {
      unsigned int start_pat = cumsum_tmp[j];
      unsigned int   end_pat = cumsum_tmp[j+1]-1;
      unsigned int size = end_pat - start_pat + 1;
      rowvec tmpcov_rstack =
        vectorise(tmpcov_tmp.rows(start_pat,end_pat),1);
      unsigned int end_tmp = tmpcov_rstack.n_elem - 1;
      cov.submat(0,0,end_tmp,end_tmp) += tmpcov_rstack.t()*tmpcov_rstack;
      // stratified cov
      set_cov[size-1] += tmpcov_rstack.t() * tmpcov_rstack;
    }
  }
  unsigned int csum_tmp = disc_ct[disc_ct.size()-1];
  cov /= static_cast<double>(csum_tmp);
  for (unsigned int i = disc_ct.size()-1; i > 0; i--) {
    cov.submat(0,0,i*(tau+1),i*(tau+1)) *=
      csum_tmp / static_cast<double>(csum_tmp+disc_ct[i-1]);
    csum_tmp += disc_ct[i-1];
  }
  // stratified cov
  for (unsigned int i = 0; i < disc_ct.size(); i++) {
    set_cov[i] /= disc_ct[i];
  }
  
  // double phi = 0;
  // meat
  usum *= -1; 
  usum.elem(find(fail==1)) += 1;
  mat Z_usum = Z.each_col() % usum; // Z % usum
  // bread
  mat Z_vsum = Z.each_col() % vsum; // Z % vsum
  mat bread_beta = Z.t() * Z_vsum;
  
  vec meat_11(m, fill::zeros),
  panmeat_11(m, fill::zeros),
  strameat_11(m, fill::zeros), 
  bread_gamma_inv(m), phi_prov(m);
  mat meat_12(m, tau+p, fill::zeros), meat_22(tau+p, tau+p, fill::zeros),
  panmeat_12(m, tau+p, fill::zeros), panmeat_22(tau+p, tau+p, fill::zeros),
  strameat_12(m,tau+p,fill::zeros), strameat_22(tau+p,tau+p,fill::zeros),
  bread_eta_gamma(tau, m), bread_beta_gamma(p, m);
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start_prov = cumsum_prov[i];
    unsigned int   end_prov = cumsum_prov[i+1]-1;
    // bread
    bread_gamma_inv(i) = 1 / accu(vsum.subvec(start_prov, end_prov));
    bread_eta_gamma.col(i) = sum(tmpmat.rows(start_prov, end_prov)).t();
    bread_beta_gamma.col(i) = sum(Z_vsum.rows(start_prov, end_prov)).t();
    // meat
    vec usum_tmp = usum.subvec(start_prov, end_prov);
    mat tmpmeat_tmp = tmpmeat.rows(start_prov, end_prov);
    mat Z_usum_tmp = Z_usum.rows(start_prov, end_prov);
    // panmeat
    mat panmeat_tmp = panmeat.rows(start_prov, end_prov); 
    mat Z_tmp = Z.rows(start_prov, end_prov);
    IntegerVector cumsum_tmp = cumsum(as<IntegerVector>(n_prov_pat[i]));
    unsigned int pats = cumsum_tmp.size();
    cumsum_tmp.push_front(0);
    for (unsigned int j = 0; j < pats; j++) {
      unsigned int start_pat = cumsum_tmp[j];
      unsigned int   end_pat = cumsum_tmp[j+1]-1;
      double u1_pat = accu(usum_tmp.subvec(start_pat, end_pat)); // for gamma_i
      meat_11(i) += u1_pat * u1_pat;
      // u2_pat: for eta and beta
      rowvec u2_pat = join_rows(sum(tmpmeat_tmp.rows(start_pat, end_pat)),
                                sum(Z_usum_tmp.rows(start_pat, end_pat)));
      meat_12.row(i) += u1_pat * u2_pat;
      meat_22 += u2_pat.t() * u2_pat;
      // phi
      // phi += pow(accu(tmp_phi.subvec(start_pat, end_pat)), 2);
      // panmeat_11
      unsigned int end_tmp = (end_pat-start_pat+1)*(tau+1)-1;
      mat cov_tmp = cov.submat(0,0,end_tmp,end_tmp);
      mat pm_pat = panmeat_tmp.rows(start_pat, end_pat);
      rowvec pm_rstack = vectorise(pm_pat, 1);
      mat pm_cov = cov_tmp.each_col()%pm_rstack.t();
      panmeat_11(i) += as_scalar(sum(pm_cov)*pm_rstack.t());
      // strameat_11
      mat stra_cov_tmp = set_cov[end_pat-start_pat];
      mat pm_stra_cov = stra_cov_tmp.each_col()%pm_rstack.t();
      strameat_11(i) += as_scalar(sum(pm_stra_cov)*pm_rstack.t());
      // panmeat_12
      mat mat_eta_stra(tau, end_tmp+1), mat_eta(tau, end_tmp+1);
      for (unsigned int k = 1; k <= tau; k++) {
        uvec idx_k = regspace<uvec>(k, tau+1, end_tmp);
        // panmeat
        mat cov_k = cov_tmp.rows(idx_k);
        mat_eta.row(k-1) = sum(cov_k.each_col()%pm_pat.col(k));
        // strameat
        mat stra_cov_k = stra_cov_tmp.rows(idx_k);
        mat_eta_stra.row(k-1) = sum(stra_cov_k.each_col()%pm_pat.col(k));
      }
      mat Z_pat = Z_tmp.rows(start_pat, end_pat);
      mat mat_beta_stra(p, end_tmp+1, fill::zeros),
          mat_beta_r(end_tmp+1, p, fill::zeros), 
      mat_beta(p, end_tmp+1, fill::zeros);
      for (unsigned int k = 0; k <= end_pat-start_pat; k++) {
        mat_beta += Z_pat.row(k).t() *
          sum(pm_cov.rows(k*(tau+1), k*(tau+1)+tau));
        // for panmeat_22
        mat_beta_r.rows(k*(tau+1), k*(tau+1)+tau) +=
          pm_pat.row(k).t() * Z_pat.row(k);
        // for strameat_22
        mat_beta_stra += Z_pat.row(k).t() *
          sum(pm_stra_cov.rows(k*(tau+1), k*(tau+1)+tau));
      }
      panmeat_12.submat(i, 0, i, tau-1) += pm_rstack * mat_eta.t();
      panmeat_12.submat(i, tau, i, tau+p-1) += pm_rstack * mat_beta.t();
      strameat_12.submat(i, 0, i, tau-1) += pm_rstack * mat_eta_stra.t();
      strameat_12.submat(i, tau, i, tau+p-1) += pm_rstack * mat_beta_stra.t();
      // panmeat_22
      for (unsigned int k = 1; k <= tau; k++) {
        uvec idx_k = regspace<uvec>(k, tau+1, end_tmp);
        panmeat_22.col(k-1) +=
          join_cols(mat_eta.cols(idx_k),mat_beta.cols(idx_k))*pm_pat.col(k);
        strameat_22.col(k-1) += join_cols(mat_eta_stra.cols(idx_k),
                        mat_beta_stra.cols(idx_k)) * pm_pat.col(k);
      }
      panmeat_22.submat(tau, tau, tau+p-1, tau+p-1) +=
        mat_beta * mat_beta_r;
      strameat_22.submat(tau, tau, tau+p-1, tau+p-1) +=
        mat_beta_stra * mat_beta_r;
    }
  }
  panmeat_22.submat(0, tau, tau-1, tau+p-1) +=
    panmeat_22.submat(tau, 0, tau+p-1, tau-1).t();
  strameat_22.submat(0, tau, tau-1, tau+p-1) +=
    strameat_22.submat(tau, 0, tau+p-1, tau-1).t();
  
  // phi /= denom_phi;
  // invert bread
  mat bread_1 = join_cols(bread_eta_gamma.each_row()%bread_gamma_inv.t(),
                          bread_beta_gamma.each_row()%bread_gamma_inv.t());
  mat schur_inv = inv_sympd(join_rows(
    join_cols(diagmat(bread_eta),bread_beta_eta),
    join_cols(bread_beta_eta.t(),bread_beta)) - 
      join_cols(bread_eta_gamma,bread_beta_gamma)*bread_1.t());
  mat bread_2 = schur_inv * bread_1;
  rowvec b_inv_idx1 = bread_1.col(idx_gamma).t() * bread_2, 
         b_inv_idx2 = bread_2.col(idx_gamma).t(); // all but a minus sign
  b_inv_idx1(idx_gamma) += bread_gamma_inv(idx_gamma);
  // standard error
  double se_str =
    sqrt(as_scalar(b_inv_idx1 * (strameat_11 % b_inv_idx1.t()) -
    2 * b_inv_idx1 * strameat_12 * b_inv_idx2.t() +
    b_inv_idx2 * strameat_22 * b_inv_idx2.t()));
  double se_mr =
    sqrt(as_scalar(b_inv_idx1 * (panmeat_11 % b_inv_idx1.t()) -
    2 * b_inv_idx1 * panmeat_12 * b_inv_idx2.t() +
    b_inv_idx2 * panmeat_22 * b_inv_idx2.t()));
  double se_r =
    sqrt(as_scalar(b_inv_idx1 * (meat_11 % b_inv_idx1.t()) -
    2 * b_inv_idx1 * meat_12 * b_inv_idx2.t() +
    b_inv_idx2 * meat_22 * b_inv_idx2.t()));
  double se_m = sqrt(b_inv_idx1(idx_gamma));
  // score
  double score =
    accu(usum.subvec(cumsum_prov[idx_gamma],cumsum_prov[idx_gamma+1]-1));
  // score test stat
  double stat_str = score * se_m * se_m / se_str; // str robust: Pan (2001b)
  double stat_mr = score * se_m * se_m / se_mr; // mod robust: Pan (2001b)
  double stat_r = score * se_m * se_m / se_r; // robust: R & J (1990)
  double stat_m = score * se_m / sqrt(phi); // model-based
  double stat_astr = score / sqrt(strameat_11(idx_gamma)); // app str
  double stat_amr = score / sqrt(panmeat_11(idx_gamma)); // app mod
  double stat_ar = score / sqrt(meat_11(idx_gamma)); // app robust
  double stat_am = score * sqrt(bread_gamma_inv(idx_gamma)); // app model
  // upper tail prob
  double prob_str = 1-normcdf(stat_str);
  double prob_mr = 1-normcdf(stat_mr);
  double prob_r = 1-normcdf(stat_r);
  double prob_m = 1-normcdf(stat_m);
  double prob_astr = 1-normcdf(stat_astr);
  double prob_amr = 1-normcdf(stat_amr);
  double prob_ar = 1-normcdf(stat_ar);
  double prob_am = 1-normcdf(stat_am);
  // flagging
  double flag_str = ifelse_scalar(prob_str < alpha/2, 1,
                                 ifelse_scalar(prob_str<=1-alpha/2, 0, -1));
  double flag_mr = ifelse_scalar(prob_mr < alpha/2, 1,
                                 ifelse_scalar(prob_mr<=1-alpha/2, 0, -1));
  double flag_r = ifelse_scalar(prob_r < alpha/2, 1, 
                                ifelse_scalar(prob_r<=1-alpha/2, 0, -1));
  double flag_m = ifelse_scalar(prob_m < alpha/2, 1, 
                                ifelse_scalar(prob_m<=1-alpha/2, 0, -1));
  double flag_astr = ifelse_scalar(prob_astr < alpha/2, 1, 
                                ifelse_scalar(prob_astr<=1-alpha/2, 0, -1));
  double flag_amr = ifelse_scalar(prob_amr < alpha/2, 1, 
                                   ifelse_scalar(prob_amr<=1-alpha/2, 0, -1));
  double flag_ar = ifelse_scalar(prob_ar < alpha/2, 1,
                                 ifelse_scalar(prob_ar<=1-alpha/2, 0, -1));
  double flag_am = ifelse_scalar(prob_am < alpha/2, 1, 
                                ifelse_scalar(prob_am<=1-alpha/2, 0, -1));
  // p-value
  double pval_str = 2*min(prob_str, 1-prob_str);
  double pval_mr = 2*min(prob_mr, 1-prob_mr);
  double pval_r = 2*min(prob_r, 1-prob_r);
  double pval_m = 2*min(prob_m, 1-prob_m);
  double pval_astr = 2*min(prob_astr, 1-prob_astr);
  double pval_amr = 2*min(prob_amr, 1-prob_amr);
  double pval_ar = 2*min(prob_ar, 1-prob_ar);
  double pval_am = 2*min(prob_am, 1-prob_am);

  NumericVector res = NumericVector::create(_["stat.strobust"]=stat_str);
  res.push_back(stat_mr, "stat.modrobust");
  res.push_back(stat_r, "stat.robust");
  res.push_back(stat_m, "stat.model");
  res.push_back(stat_astr, "stat.astrobust");
  res.push_back(stat_amr, "stat.amodrobust");
  res.push_back(stat_ar, "stat.arobust");
  res.push_back(stat_am, "stat.amodel");
  res.push_back(pval_str, "p.strobust");
  res.push_back(pval_mr, "p.modrobust");
  res.push_back(pval_r, "p.robust");
  res.push_back(pval_m, "p.model");
  res.push_back(pval_astr, "p.astrobust");
  res.push_back(pval_amr, "p.amodrobust");
  res.push_back(pval_ar, "p.arobust");
  res.push_back(pval_am, "p.amodel");
  res.push_back(flag_str, "flag.strobust");
  res.push_back(flag_mr, "flag.modrobust");
  res.push_back(flag_r, "flag.robust");
  res.push_back(flag_m, "flag.model");
  res.push_back(flag_astr, "flag.astrobust");
  res.push_back(flag_amr, "flag.amodrobust");
  res.push_back(flag_ar, "flag.arobust");
  res.push_back(flag_am, "flag.amodel");
  res.push_back(se_str, "se.strobust");
  res.push_back(se_mr, "se.modrobust");
  res.push_back(se_r, "se.robust");
  res.push_back(se_m, "se.model");
  res.push_back(phi, "scale");
  return res;
}

// [[Rcpp::export]]
List bin_logit_cr_wald(vec &fail, mat &Z, IntegerVector &n_prov, 
                                 List n_prov_pat, vec &time, vec gamma, vec eta, 
                                 vec beta, double alpha=0.05) {

  double gamma_med = median(gamma);
  unsigned int m = gamma.n_elem, n = Z.n_rows; // prov/sample count
  unsigned int p = beta.n_elem, tau = eta.n_elem; // covar/time count
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vec uniqtime = unique(time), obs_time(tau+1); // count by time
  vector<uvec> idx_time; // index of disc stratified by time
  int denom_phi = -(m + tau + p); // denominator of phi
  for (unsigned int i = 0; i <= tau; i++) {
    uvec idx_tmp = find(time == uniqtime(i));
    idx_time.push_back(idx_tmp);
    obs_time(i) = accu(fail.elem(idx_tmp));
    denom_phi += (i + 1) * idx_tmp.n_elem;
  }
  vec obs_prov(m); // failure count by provider
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
  }
  vec tmp = gamma_provtodis(gamma, cumsum_prov) + Z * beta;
  vec oddsum = exp(tmp); // h/(1-h) for squared Pearson resid
  vec oddsum_t0 = oddsum.elem(idx_time[0]);
  vec oddsum_t0_fail = oddsum_t0.elem(find(fail.elem(idx_time[0])==1));
  double phi = accu(1/oddsum_t0_fail - oddsum_t0_fail); // sum of (1-2h)/[h(1-h)]
  // sum of u_{ij}(k) or h'_{ij}(k) over k for each dis
  vec usum = 1 / (1 + 1/oddsum); // for meat, init at time k = 1
  vec vsum = usum % (1 - usum); // for bread, init at time k = 1
  // vec tmp_phi = -sqrt(oddsum);
  // tmp_phi.elem(idx_time[0]) += 
  //   fail.elem(idx_time[0]) / sqrt(vsum.elem(idx_time[0]));
  vec bread_eta(tau, fill::zeros);
  mat bread_beta_eta(p, tau, fill::zeros);
  mat tmpcov(n, tau+1, fill::zeros); // for cov matrix, see Pan(2001b)
  tmpcov.col(0) -= sqrt(oddsum);
  uvec col_cov(1); col_cov.fill(0);
  tmpcov.submat(idx_time[0], col_cov) +=
    fail.elem(idx_time[0]) / sqrt(vsum.elem(idx_time[0]));
  mat tmpmeat(n, tau, fill::zeros); // for meat
  mat panmeat(n, tau+1, fill::zeros); // for meat by Pan (2001b)
  panmeat.col(0) = sqrt(vsum);
  mat tmp_pm(n, tau, fill::zeros); // for meat by Pan (2001b)
  mat tmpmat(n, tau, fill::zeros); // for bread
  for (unsigned int i = 1; i <= tau; i++) {
    mat Z_tmp = Z.rows(idx_time[i]); // for bread
    uvec col_meat(1); col_meat.fill(i-1);
    tmpmeat.submat(idx_time[i], col_meat) += fail.elem(idx_time[i]);
    for (unsigned int k = 0; k < tau; k++) {
      vec odds_tmp = exp(eta(k)+tmp.elem(idx_time[i]));
      vec h_tmp = 1 / (1 + 1/odds_tmp);
      vec v_tmp = h_tmp % (1 - h_tmp);
      uvec col(1); col.fill(k);
      // for panmeat
      panmeat.submat(idx_time[i], col+1) += sqrt(v_tmp);
      if (k < i) { // at risk
        // meat
        usum.elem(idx_time[i]) += h_tmp;
        tmpmeat.submat(idx_time[i], col) -= h_tmp;
        // bread
        vsum.elem(idx_time[i]) += v_tmp;
        bread_eta(k) += accu(v_tmp);
        tmpmat.submat(idx_time[i], col) += v_tmp;
        bread_beta_eta.col(k) += sum(Z_tmp.each_col() % v_tmp).t();
        // phi
        oddsum.elem(idx_time[i]) += odds_tmp;
        // tmp_phi.elem(idx_time[i]) -= sqrt(odds_tmp);
        // cov
        tmpcov.submat(idx_time[i], col+1) -= sqrt(odds_tmp);
        if (k==i-1) {
          // tmp_phi.elem(idx_time[i]) += fail.elem(idx_time[i])/sqrt(v_tmp);
          vec odds_temp = odds_tmp.elem(find(fail.elem(idx_time[i])==1));
          phi += accu(1/odds_temp - odds_temp);
          // cov
          tmpcov.submat(idx_time[i], col+1) += 
            fail.elem(idx_time[i]) / sqrt(v_tmp);
          // panmeat
          tmp_pm.submat(idx_time[i], col) += sqrt(v_tmp);
        }        
      }
    }
  }
  // phi
  phi += accu(oddsum);
  phi /= denom_phi;
  // cov
  unsigned int dim_cov = max(unlist(n_prov_pat))*(tau+1);
  // stratified cov
  IntegerVector disc_ct = table(unlist(n_prov_pat));
  vector<mat> set_cov;
  for (unsigned int i = 0; i < disc_ct.size(); i++) {
    mat cov_mat_tmp((i+1)*(tau+1), (i+1)*(tau+1), fill::zeros);
    set_cov.push_back(cov_mat_tmp);
  }
  mat cov(dim_cov, dim_cov, fill::zeros);
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start_prov = cumsum_prov[i];
    unsigned int   end_prov = cumsum_prov[i+1]-1;
    mat tmpcov_tmp = tmpcov.rows(start_prov, end_prov);
    IntegerVector cumsum_tmp = cumsum(as<IntegerVector>(n_prov_pat[i]));
    unsigned int pats = cumsum_tmp.size(); // patient ct
    
    cumsum_tmp.push_front(0);
    for (unsigned int j = 0; j < pats; j++) {
      unsigned int start_pat = cumsum_tmp[j];
      unsigned int   end_pat = cumsum_tmp[j+1]-1;
      unsigned int size = end_pat - start_pat + 1;
      rowvec tmpcov_rstack =
        vectorise(tmpcov_tmp.rows(start_pat,end_pat),1);
      unsigned int end_tmp = tmpcov_rstack.n_elem - 1;
      cov.submat(0,0,end_tmp,end_tmp) += tmpcov_rstack.t()*tmpcov_rstack;
      // stratified cov
      set_cov[size-1] += tmpcov_rstack.t() * tmpcov_rstack;
    }
  }
  unsigned int csum_tmp = disc_ct[disc_ct.size()-1];
  cov /= static_cast<double>(csum_tmp);
  for (unsigned int i = disc_ct.size()-1; i > 0; i--) {
    cov.submat(0,0,i*(tau+1),i*(tau+1)) *=
      csum_tmp / static_cast<double>(csum_tmp+disc_ct[i-1]);
    csum_tmp += disc_ct[i-1];
  }
  // stratified cov
  for (unsigned int i = 0; i < disc_ct.size(); i++) {
    set_cov[i] /= disc_ct[i];
  }
  
  // double phi = 0;
  // meat
  usum *= -1; 
  usum.elem(find(fail==1)) += 1;
  mat Z_usum = Z.each_col() % usum; // Z % usum
  // bread
  mat Z_vsum = Z.each_col() % vsum; // Z % vsum
  mat bread_beta = Z.t() * Z_vsum;
  
  vec meat_11(m, fill::zeros),
  panmeat_11(m, fill::zeros),
  strameat_11(m, fill::zeros), 
  bread_gamma_inv(m), phi_prov(m);
  mat meat_12(m, tau+p, fill::zeros), meat_22(tau+p, tau+p, fill::zeros),
  panmeat_12(m, tau+p, fill::zeros), panmeat_22(tau+p, tau+p, fill::zeros),
  strameat_12(m,tau+p,fill::zeros), strameat_22(tau+p,tau+p,fill::zeros),
  bread_eta_gamma(tau, m), bread_beta_gamma(p, m);
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start_prov = cumsum_prov[i];
    unsigned int   end_prov = cumsum_prov[i+1]-1;
    // bread
    bread_gamma_inv(i) = 1 / accu(vsum.subvec(start_prov, end_prov));
    bread_eta_gamma.col(i) = sum(tmpmat.rows(start_prov, end_prov)).t();
    bread_beta_gamma.col(i) = sum(Z_vsum.rows(start_prov, end_prov)).t();
    // meat
    vec usum_tmp = usum.subvec(start_prov, end_prov);
    mat tmpmeat_tmp = tmpmeat.rows(start_prov, end_prov);
    mat Z_usum_tmp = Z_usum.rows(start_prov, end_prov);
    // panmeat
    mat panmeat_tmp = panmeat.rows(start_prov, end_prov); 
    mat Z_tmp = Z.rows(start_prov, end_prov);
    IntegerVector cumsum_tmp = cumsum(as<IntegerVector>(n_prov_pat[i]));
    unsigned int pats = cumsum_tmp.size();
    cumsum_tmp.push_front(0);
    for (unsigned int j = 0; j < pats; j++) {
      unsigned int start_pat = cumsum_tmp[j];
      unsigned int   end_pat = cumsum_tmp[j+1]-1;
      double u1_pat = accu(usum_tmp.subvec(start_pat, end_pat)); // for gamma_i
      meat_11(i) += u1_pat * u1_pat;
      // u2_pat: for eta and beta
      rowvec u2_pat = join_rows(sum(tmpmeat_tmp.rows(start_pat, end_pat)),
                                sum(Z_usum_tmp.rows(start_pat, end_pat)));
      meat_12.row(i) += u1_pat * u2_pat;
      meat_22 += u2_pat.t() * u2_pat;
      // phi
      // phi += pow(accu(tmp_phi.subvec(start_pat, end_pat)), 2);
      // panmeat_11
      unsigned int end_tmp = (end_pat-start_pat+1)*(tau+1)-1;
      mat cov_tmp = cov.submat(0,0,end_tmp,end_tmp);
      mat pm_pat = panmeat_tmp.rows(start_pat, end_pat);
      rowvec pm_rstack = vectorise(pm_pat, 1);
      mat pm_cov = cov_tmp.each_col()%pm_rstack.t();
      panmeat_11(i) += as_scalar(sum(pm_cov)*pm_rstack.t());
      // strameat_11
      mat stra_cov_tmp = set_cov[end_pat-start_pat];
      mat pm_stra_cov = stra_cov_tmp.each_col()%pm_rstack.t();
      strameat_11(i) += as_scalar(sum(pm_stra_cov)*pm_rstack.t());
      // panmeat_12
      mat mat_eta_stra(tau, end_tmp+1), mat_eta(tau, end_tmp+1);
      for (unsigned int k = 1; k <= tau; k++) {
        uvec idx_k = regspace<uvec>(k, tau+1, end_tmp);
        // panmeat
        mat cov_k = cov_tmp.rows(idx_k);
        mat_eta.row(k-1) = sum(cov_k.each_col()%pm_pat.col(k));
        // strameat
        mat stra_cov_k = stra_cov_tmp.rows(idx_k);
        mat_eta_stra.row(k-1) = sum(stra_cov_k.each_col()%pm_pat.col(k));
      }
      mat Z_pat = Z_tmp.rows(start_pat, end_pat);
      mat mat_beta_stra(p, end_tmp+1, fill::zeros),
      mat_beta_r(end_tmp+1, p, fill::zeros), 
      mat_beta(p, end_tmp+1, fill::zeros);
      for (unsigned int k = 0; k <= end_pat-start_pat; k++) {
        mat_beta += Z_pat.row(k).t() *
          sum(pm_cov.rows(k*(tau+1), k*(tau+1)+tau));
        // for panmeat_22
        mat_beta_r.rows(k*(tau+1), k*(tau+1)+tau) +=
          pm_pat.row(k).t() * Z_pat.row(k);
        // for strameat_22
        mat_beta_stra += Z_pat.row(k).t() *
          sum(pm_stra_cov.rows(k*(tau+1), k*(tau+1)+tau));
      }
      panmeat_12.submat(i, 0, i, tau-1) += pm_rstack * mat_eta.t();
      panmeat_12.submat(i, tau, i, tau+p-1) += pm_rstack * mat_beta.t();
      strameat_12.submat(i, 0, i, tau-1) += pm_rstack * mat_eta_stra.t();
      strameat_12.submat(i, tau, i, tau+p-1) += pm_rstack * mat_beta_stra.t();
      // panmeat_22
      for (unsigned int k = 1; k <= tau; k++) {
        uvec idx_k = regspace<uvec>(k, tau+1, end_tmp);
        panmeat_22.col(k-1) +=
          join_cols(mat_eta.cols(idx_k),mat_beta.cols(idx_k))*pm_pat.col(k);
        strameat_22.col(k-1) += join_cols(mat_eta_stra.cols(idx_k),
                        mat_beta_stra.cols(idx_k)) * pm_pat.col(k);
      }
      panmeat_22.submat(tau, tau, tau+p-1, tau+p-1) +=
        mat_beta * mat_beta_r;
      strameat_22.submat(tau, tau, tau+p-1, tau+p-1) +=
        mat_beta_stra * mat_beta_r;
    }
  }
  panmeat_22.submat(0, tau, tau-1, tau+p-1) +=
    panmeat_22.submat(tau, 0, tau+p-1, tau-1).t();
  strameat_22.submat(0, tau, tau-1, tau+p-1) +=
    strameat_22.submat(tau, 0, tau+p-1, tau-1).t();
  
  // phi /= denom_phi;
  // invert bread
  mat bread_1 = join_cols(bread_eta_gamma.each_row()%bread_gamma_inv.t(),
                          bread_beta_gamma.each_row()%bread_gamma_inv.t());
  mat schur_inv = inv_sympd(join_rows(
    join_cols(diagmat(bread_eta),bread_beta_eta),
    join_cols(bread_beta_eta.t(),bread_beta)) - 
      join_cols(bread_eta_gamma,bread_beta_gamma)*bread_1.t());
  mat bread_2 = schur_inv * bread_1;
  vec se_str(m), se_mr(m), se_r(m), se_m(m);
  // standard errors
  for (unsigned int i = 0; i < m; i++) {
    rowvec b_inv_idx1 = bread_1.col(i).t() * bread_2, 
      b_inv_idx2 = bread_2.col(i).t(); // all but a minus sign
    b_inv_idx1(i) += bread_gamma_inv(i);
    se_str(i) =
      sqrt(as_scalar(b_inv_idx1 * (strameat_11 % b_inv_idx1.t()) -
      2 * b_inv_idx1 * strameat_12 * b_inv_idx2.t() +
      b_inv_idx2 * strameat_22 * b_inv_idx2.t()));
    se_mr(i) =
      sqrt(as_scalar(b_inv_idx1 * (panmeat_11 % b_inv_idx1.t()) -
      2 * b_inv_idx1 * panmeat_12 * b_inv_idx2.t() +
      b_inv_idx2 * panmeat_22 * b_inv_idx2.t()));
    se_r(i) =
      sqrt(as_scalar(b_inv_idx1 * (meat_11 % b_inv_idx1.t()) -
      2 * b_inv_idx1 * meat_12 * b_inv_idx2.t() +
      b_inv_idx2 * meat_22 * b_inv_idx2.t()));
    se_m(i) = sqrt(b_inv_idx1(i));
  }
  mat se(m, 4);
  se.col(0)=se_str; se.col(1)=se_mr; se.col(2)=se_r; se.col(3)=se_m;
  vec stat_numer = gamma - gamma_med;
  // wald test stat
  vec stat_str = stat_numer / se_str,
    stat_mr = stat_numer / se_mr,
    stat_r = stat_numer / se_r,
    stat_m = stat_numer / se_m / sqrt(phi),
    stat_astr = stat_numer / bread_gamma_inv / sqrt(strameat_11),
    stat_amr = stat_numer / bread_gamma_inv / sqrt(panmeat_11),
    stat_ar = stat_numer / bread_gamma_inv / sqrt(meat_11),
    stat_am = stat_numer / sqrt(bread_gamma_inv); // app model
  mat stat(m, 8);
  stat.col(0) = stat_str; stat.col(1) = stat_mr;
  stat.col(2) = stat_r; stat.col(3) = stat_m;
  stat.col(4) = stat_astr; stat.col(5) = stat_amr;
  stat.col(6) = stat_ar; stat.col(7) = stat_am;
  // upper tail prob
  mat prob = 1-normcdf(stat);
  // flagging
  imat flag(m, 8, fill::zeros);
  flag.elem(find(prob < alpha/2)).ones();
  flag.elem(find(prob > 1-alpha/2)).fill(-1);
  // p-values
  mat pval = 2 * min(prob, 1-prob);
  List res = List::create(_["stat"]=stat, _["p"]=pval, _["flag"]=flag, 
                          _["se"]=se, _["scale"]=phi);
  return res;
}

// [[Rcpp::export]]
vec SRR_bin_logit_cr(vec &gamma, vec &eta, vec &beta, 
                     IntegerVector &n_prov, vec &fail, vec &time, mat &Z) {
  unsigned int m = gamma.n_elem, tau = eta.n_elem;
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vector<uvec> idx_time; // index of disc stratified by time
  vec uniqtime = unique(time);
  for (unsigned int i = 0; i <= tau; i++) {
    idx_time.push_back(find(time == uniqtime(i)));
  }
  vec tmp = gamma_provtodis(median(gamma)*ones(m), cumsum_prov) + Z*beta;
  vec hazsum = 1 / (1 + exp(-tmp)); // initialized at time k = 1
  for (unsigned int i = 1; i <= tau; i++) {
    for (unsigned int k = 0; k < i; k++) { // at risk
      hazsum.elem(idx_time[i]) += 
        1 / (1+exp(-eta(k)-tmp.elem(idx_time[i])));
    }
  }
  vec obs_prov(m), exp_prov(m);
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
    exp_prov(i) = accu(hazsum.subvec(start, end));
  }
  return obs_prov / exp_prov;
}

// [[Rcpp::export]]
List bin_log_cr(vec &fail, mat &Z, IntegerVector &n_prov, vec &time,
                vec gamma, vec eta, vec beta, 
                int parallel=1, int threads=1,
                double s=0.01, double t=0.6, 
                double tol=1e-8, int max_iter=100, double bound=10.0) {
  
  unsigned int m = gamma.n_elem, n = Z.n_rows; // prov/sample count
  unsigned int p = beta.n_elem, tau = eta.n_elem; // covar/time count
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vec uniqtime = unique(time);
  vector<uvec> idx_time; // index of disc stratified by time
  for (unsigned int i = 0; i <= tau; i++) {
    uvec idx_tmp = find(time == uniqtime(i));
    idx_time.push_back(idx_tmp);
  }
  vec obs_prov(m); // failure count by provider
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
  }
  double d_obj, v, inc, crit = 100.0;
  double obj = Loglkd_log(fail, Z, cumsum_prov, idx_time, 
                          gamma, eta, beta);// initialize objetive function
  int iter = 0;
  uvec idx_fail = find(fail==1);
  mat Z_fail = Z.rows(idx_fail);
  cout << "Implementing BIN (log) ..." << endl;
  while (iter < max_iter && crit >= tol) {
    iter++;
    // calculating scores and info matrices
    vec tmp = gamma_provtodis(gamma, cumsum_prov) + Z * beta;
    vec u(n), w(n); // u_{ij}(t_ij) and w_{ij}(t_ij) for each dis
    // sum of u_{ij}(k) or w_{ij}(k) over k for each dis
    vec ucum = 1 / (exp(-tmp)-1); // initialized at time k = 1
    vec icum = ucum % (1 + ucum); // initialized at time k = 1
    u.elem(idx_time[0]) = ucum.elem(idx_time[0]);
    w.elem(idx_time[0]) = icum.elem(idx_time[0]);
    vec score_eta(tau, fill::zeros), info_eta(tau, fill::zeros);
    mat info_beta_eta(p, tau, fill::zeros);
    mat tmpmat(n, tau, fill::zeros); // tmpmat to info_eta_gamma
    for (unsigned int i = 1; i <= tau; i++) {
      mat Z_tmp = Z.rows(idx_time[i]);
      vec fail_tmp = fail.elem(idx_time[i]);
      for (unsigned int k = 0; k < i; k++) { // at risk
        vec u_tmp = 1 / (exp(-eta(k)-tmp.elem(idx_time[i]))-1);
        vec i_tmp = u_tmp % (1 + u_tmp);
        ucum.elem(idx_time[i]) += u_tmp;
        icum.elem(idx_time[i]) += i_tmp;
        uvec col_tmpmat(1); col_tmpmat.fill(k);
        if (k==i-1) {
          uvec idx_tmp = find(fail_tmp==1);
          u.elem(idx_time[i]) = u_tmp;
          w.elem(idx_time[i]) = i_tmp;
          score_eta(k) += accu(1 + u_tmp.elem(idx_tmp));
          info_eta(k) -= accu(i_tmp.elem(idx_tmp));
          tmpmat.submat(idx_time[i], col_tmpmat) -= fail_tmp % i_tmp;
          info_beta_eta.col(k) -= 
            sum(Z_tmp.each_col() % (fail_tmp % i_tmp)).t();
        }
        score_eta(k) -= accu(u_tmp);
        info_eta(k) += accu(i_tmp);
        tmpmat.submat(idx_time[i], col_tmpmat) += i_tmp;
        info_beta_eta.col(k) += sum(Z_tmp.each_col() % i_tmp).t();
        // note: a non-contiguous submatrix doesn't have each_col as 
        //       its member function
      }
    }
    mat Z_sum = Z.each_col() % ucum; // Z % vsum
    vec score_beta = sum(Z_fail.each_col() % (1 + w.elem(idx_fail))).t()
      - sum(Z_sum).t();
    Z_sum = Z.each_col() % (icum - (fail % w)); // Z % (icum - (fail % w))
    mat info_beta = Z.t() * Z_sum;
    vec score_gamma(m), info_gamma_inv(m);
    mat info_eta_gamma(tau, m), info_beta_gamma(p, m);
    for (unsigned int i = 0; i < m; i++) {
      unsigned int start = cumsum_prov[i]; 
      unsigned int   end = cumsum_prov[i+1]-1;
      score_gamma(i) = 
        accu(fail.subvec(start, end) % (1+u.subvec(start, end)))
        - accu(ucum.subvec(start, end));
      info_gamma_inv(i) = 1 / 
      (accu(icum.subvec(start, end))
         - accu(fail.subvec(start, end) % w.subvec(start, end)));
      info_eta_gamma.col(i) = sum(tmpmat.rows(start, end)).t();
      info_beta_gamma.col(i) = sum(Z_sum.rows(start, end)).t();
    }
    // Newton step
    mat info_1 = join_cols(info_eta_gamma.each_row()%info_gamma_inv.t(),
                           info_beta_gamma.each_row()%info_gamma_inv.t());
    mat schur_inv = inv_sympd(join_rows(
      join_cols(diagmat(info_eta),info_beta_eta),
      join_cols(info_beta_eta.t(),info_beta)) - 
        join_cols(info_eta_gamma,info_beta_gamma)*info_1.t());
    mat info_2 = schur_inv * info_1;
    vec d_gamma = info_gamma_inv % score_gamma + 
      info_2.t()*(info_1*score_gamma-join_cols(score_eta,score_beta));
    vec d_etabeta = schur_inv*join_cols(score_eta,score_beta) - 
      info_2*score_gamma;
    vec d_eta = d_etabeta.head(tau), d_beta = d_etabeta.tail(p);
    // backtracking line search
    v = 1.0; // initial step size
    d_obj = Loglkd_log(fail, Z, cumsum_prov, idx_time, 
                       gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    inc = dot(score_gamma, d_gamma) + dot(score_eta, d_eta) +
      dot(score_beta, d_beta); // square of Newton increment
    while (d_obj < s*v*inc) {
      v *= t;
      d_obj = Loglkd_log(fail, Z, cumsum_prov, idx_time, 
                         gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    }
    gamma += v * d_gamma;
    double med_gamma = median(gamma);
    gamma = clamp(gamma, med_gamma-bound, med_gamma+bound);
    beta += v * d_beta;
    eta += v * d_eta;
    obj += d_obj;
    
    crit = v * norm(d_etabeta, "inf");
    cout << "Iter " << iter << ": loglkd = " << scientific 
         << setprecision(5) << obj << ";";
    cout << " running diff = " << scientific
         << setprecision(5) << crit << ";" << endl;
  }
  // calculate SRR
  vec tmp = gamma_provtodis(median(gamma)*ones(m), cumsum_prov) + Z*beta;
  vec hazsum = exp(tmp); // initialized at time k = 1
  for (unsigned int i = 1; i <= tau; i++) {
    for (unsigned int k = 0; k < i; k++) { // at risk
      hazsum.elem(idx_time[i]) += exp(eta(k)+tmp.elem(idx_time[i]));
    }
  }
  vec exp_prov(m); // denominator of SRR
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    exp_prov(i) = accu(hazsum.subvec(start, end));
  }
  cout << "BIN converged after " << iter << " iterations!" << endl;
  List ret = List::create(_["gamma"]=gamma, 
                          _["eta"]=eta, _["beta"]=beta,
                          _["Exp"]=exp_prov,
                          _["SRR"]=obs_prov/exp_prov);
  return ret;
}

// [[Rcpp::export]]
List bin_cloglog_cr(vec &fail, mat &Z, IntegerVector &n_prov, vec &time,
                    vec gamma, vec eta, vec beta, 
                    int parallel=1, int threads=1,
                    double s=0.01, double t=0.6, 
                    double tol=1e-8, int max_iter=100, double bound=10.0) {
  
  unsigned int m = gamma.n_elem, n = Z.n_rows; // prov/sample count
  unsigned int p = beta.n_elem, tau = eta.n_elem; // covar/time count
  IntegerVector cumsum_prov = cumsum(n_prov);
  cumsum_prov.push_front(0);
  vec uniqtime = unique(time);
  vector<uvec> idx_time; // index of disc stratified by time
  for (unsigned int i = 0; i <= tau; i++) {
    uvec idx_tmp = find(time == uniqtime(i));
    idx_time.push_back(idx_tmp);
  }
  double d_obj, v, inc, crit = 100.0;
  double obj = Loglkd_cloglog(fail, Z, cumsum_prov, idx_time, 
                              gamma, eta, beta); // init obj fun
  int iter = 0;
  uvec idx_fail = find(fail==1);
  mat Z_fail = Z.rows(idx_fail);
  cout << "Implementing BIN (cloglog) ..." << endl;
  while (iter < max_iter && crit >= tol) {
    iter++;
    // calculating scores and info matrices
    vec tmp = gamma_provtodis(gamma, cumsum_prov) + Z * beta;
    vec uear(n), iear(n); // part of u or i at end of at-risk time
    vec ucum = -exp(tmp); // log(1-h)
    // log(1-h) / h
    vec uh = ucum.elem(idx_time[0])/(1-exp(ucum.elem(idx_time[0])));
    uear.elem(idx_time[0]) = uh; // log(1-h) / h
    iear.elem(idx_time[0]) = uh + exp(uh) % uh % uh;
    vec score_eta(tau, fill::zeros), info_eta(tau, fill::zeros);
    mat info_beta_eta(p, tau, fill::zeros);
    mat tmpmat(n, tau, fill::zeros); // tmpmat to info_eta_gamma
    for (unsigned int i = 1; i <= tau; i++) {
      mat Z_tmp = Z.rows(idx_time[i]);
      vec fail_tmp = fail.elem(idx_time[i]);
      for (unsigned int k = 0; k < i; k++) { // at risk
        vec u_tmp = -exp(eta(k)+tmp.elem(idx_time[i])); // log(1-h)
        ucum.elem(idx_time[i]) += u_tmp;
        uvec col_tmpmat(1); col_tmpmat.fill(k);
        tmpmat.submat(idx_time[i], col_tmpmat) -= u_tmp;
        info_beta_eta.col(k) -= sum(Z_tmp.each_col() % u_tmp).t();
        double sum_u_tmp = accu(u_tmp);
        score_eta(k) += sum_u_tmp;
        info_eta(k) -= sum_u_tmp;
        if (k==i-1) {
          uvec idx_tmp = find(fail_tmp==1);
          vec uh = u_tmp / (1 - exp(u_tmp));
          uear.elem(idx_time[i]) = uh;
          vec vec_tmp = uh + exp(uh) % uh % uh;
          iear.elem(idx_time[i]) = vec_tmp;
          score_eta(k) -= accu(uh.elem(idx_tmp));
          info_eta(k) += accu(vec_tmp.elem(idx_tmp));
          tmpmat.submat(idx_time[i], col_tmpmat) += fail_tmp % vec_tmp;
          info_beta_eta.col(k) +=
            sum(Z_tmp.each_col() % (fail_tmp%vec_tmp)).t();
        }
      }
    }
    mat Z_sum = Z.each_col() % ucum; // Z % ucum
    vec score_beta = sum(Z_sum).t()
      - sum(Z_fail.each_col() % uear.elem(idx_fail)).t();
    Z_sum = Z.each_col() % ((fail % iear) - ucum);
    mat info_beta = Z.t() * Z_sum;
    vec score_gamma(m), info_gamma_inv(m);
    mat info_eta_gamma(tau, m), info_beta_gamma(p, m);
    for (unsigned int i = 0; i < m; i++) {
      unsigned int start = cumsum_prov[i]; 
      unsigned int   end = cumsum_prov[i+1]-1;
      score_gamma(i) = accu(ucum.subvec(start, end))
        - accu(fail.subvec(start, end) % uear.subvec(start, end));
      info_gamma_inv(i) = 1 /
        (accu(fail.subvec(start, end) % iear.subvec(start, end))
           - accu(ucum.subvec(start, end)));
      info_eta_gamma.col(i) = sum(tmpmat.rows(start, end)).t();
      info_beta_gamma.col(i) = sum(Z_sum.rows(start, end)).t();
    }
    // Newton step
    mat info_1 = join_cols(info_eta_gamma.each_row()%info_gamma_inv.t(),
                           info_beta_gamma.each_row()%info_gamma_inv.t());
    mat schur_inv = inv_sympd(join_rows(
      join_cols(diagmat(info_eta),info_beta_eta),
      join_cols(info_beta_eta.t(),info_beta)) - 
        join_cols(info_eta_gamma,info_beta_gamma)*info_1.t());
    mat info_2 = schur_inv * info_1;
    vec d_gamma = info_gamma_inv % score_gamma + 
      info_2.t()*(info_1*score_gamma-join_cols(score_eta,score_beta));
    vec d_etabeta = schur_inv*join_cols(score_eta,score_beta) - 
      info_2*score_gamma;
    vec d_eta = d_etabeta.head(tau), d_beta = d_etabeta.tail(p);
    // backtracking line search
    v = 1.0; // initial step size
    d_obj = Loglkd_cloglog(fail, Z, cumsum_prov, idx_time, 
                           gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    inc = dot(score_gamma, d_gamma) + dot(score_eta, d_eta) +
      dot(score_beta, d_beta); // square of Newton increment
    while (d_obj < s*v*inc) {
      v *= t;
      d_obj = Loglkd_cloglog(fail, Z, cumsum_prov, idx_time, 
                             gamma+v*d_gamma, eta+v*d_eta, beta+v*d_beta) - obj;
    }
    gamma += v * d_gamma;
    double med_gamma = median(gamma);
    gamma = clamp(gamma, med_gamma-bound, med_gamma+bound);
    beta += v * d_beta;
    eta += v * d_eta;
    obj += d_obj;
    
    crit = v * norm(d_etabeta, "inf");
    cout << "Iter " << iter << ": loglkd = " << scientific 
         << setprecision(5) << obj << ";";
    cout << " running diff = " << scientific
         << setprecision(5) << crit << ";" << endl;
  }
  // calculate SRR
  vec tmp = gamma_provtodis(median(gamma)*ones(m), cumsum_prov) + Z*beta;
  vec hazsum = 1 - exp(-exp(tmp)); // initialized at time k = 1
  for (unsigned int i = 1; i <= tau; i++) {
    for (unsigned int k = 0; k < i; k++) { // at risk
      hazsum.elem(idx_time[i]) += 
        1 - exp(-exp(eta(k)+tmp.elem(idx_time[i])));
    }
  }
  // numer/denom of SRR
  vec obs_prov(m), exp_prov(m); 
  for (unsigned int i = 0; i < m; i++) {
    unsigned int start = cumsum_prov[i]; 
    unsigned int   end = cumsum_prov[i+1]-1;
    obs_prov(i) = accu(fail.subvec(start, end));
    exp_prov(i) = accu(hazsum.subvec(start, end));
  }
  cout << "BIN converged after " << iter << " iterations!" << endl;
  List ret = List::create(_["gamma"]=gamma, 
                          _["eta"]=eta, _["beta"]=beta,
                          _["Exp"]=exp_prov,
                          _["SRR"]=obs_prov/exp_prov);
  return ret;
}