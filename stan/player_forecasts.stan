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
    for (i in 1:nplayer) theta[i] ~ dirichlet(alpha);
    for (i in 1:nplayer) {
        for (j in 1:nmatch) {
            y[i, j] ~ multinomial(theta_mins[i, j]);
        }
    }
}