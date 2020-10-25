data {
    int<lower=1> nplayer;
    int<lower=1> nmatch;
    matrix[nplayer, nmatch] minutes;
    int y[nplayer, nmatch, 3];
    vector[3] alpha;
}
parameters {
    simplex[3] theta[nplayer];
}
transformed parameters {
    // construct the modified simplex that accounts for minutes played
    simplex[3] theta_mins[nplayer, nmatch];
    for (i in 1:nplayer) {
        for (j in 1:nmatch) {
            theta_mins[i, j][1] = theta[i][1] * (minutes[i, j] / 90);
            theta_mins[i, j][2] = theta[i][2] * (minutes[i, j] / 90);
            theta_mins[i, j][3] = theta[i][3] * (minutes[i, j] / 90) + (90 - minutes[i, j]) / 90;
        }
    }
}
model {
    //vector[3] alpha = rep_vector(1.0, 3); think if we can do better than this
    //alpha ~ lognormal(0, 1);
    for (i in 1:nplayer) theta[i] ~ dirichlet(alpha);
    for (i in 1:nplayer) {
        for (j in 1:nmatch) {
            y[i, j] ~ multinomial(theta_mins[i, j]);
        }
    }
}
generated quantities {
    int y_rep[nplayer, nmatch, 3];
    for (i in 1:nplayer) {
        for (j in 1:nmatch) {
            int goals = sum(y[i, j]);
            if (goals > 0) {
                y_rep[i, j] = multinomial_rng(theta_mins[i, j], goals);
            }
            else {
                y_rep[i, j] = rep_array(0, 3);
            }
        }
    }
}