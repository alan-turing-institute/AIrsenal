data {
  int n_match;
  int n_team;
  int n_gameweeks;
  int home_goals[n_match];
  int away_goals[n_match];
  int home_team[n_match];
  int away_team[n_match];
  int gameweek[n_match];
}
parameters {
  matrix[n_team, n_gameweeks] a_std;
  matrix[n_team, n_gameweeks] b_std;
  real<lower=0> sigma_ar;
  real<lower=0> sigma;
  real mu_b;
  real<lower=0> gamma;
}
transformed parameters {
  matrix[n_team, n_gameweeks] a;
  matrix[n_team, n_gameweeks] b;

  for (i in 1:n_team) {
      a[i, 1] = sigma * a_std[i, 1];
      b[i, 1] = mu_b + sigma * b_std[i, 1];

      for (j in 2:n_gameweeks) {
        a[i, j] = a[i, j - 1] + sigma_ar * a_std[i, j];
        b[i, j] = b[i, j - 1] + sigma_ar * b_std[i, j];
      }
  }
}
model {
    // scoring rates
    vector[n_match] home_rate;
    vector[n_match] away_rate;
    for (i in 1:n_match) {
        int home_idx = home_team[i];
        int away_idx = away_team[i];
        int gameweek_idx = gameweek[i];
        home_rate[i] = a[home_idx, gameweek_idx] + b[away_idx, gameweek_idx] + log(gamma);
        away_rate[i] = a[away_idx, gameweek_idx] + b[home_idx, gameweek_idx];
    }

    to_vector(a_std) ~ normal(0, 1);
    to_vector(b_std) ~ normal(0, 1);
    sigma_ar ~ normal(0, 0.1);
    sigma ~ normal(0, 1);
    mu_b ~ normal(0, 1);
    gamma ~ normal(1.4, 0.3);

    home_goals ~ poisson_log(home_rate);
    away_goals ~ poisson_log(away_rate);
}
