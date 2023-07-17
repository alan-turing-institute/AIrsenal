data {
    int<lower=1> nmatch;
    vector[nmatch] minutes;
    int y[nmatch, 3];
}
parameters {
    simplex[3] theta;
}
transformed parameters {
    // construct the modified simplex that accounts for minutes played
    simplex[3] theta_mins[nmatch];
    for (j in 1:nmatch) {
            theta_mins[j][1] = theta[1] * (minutes[j] / 90);
            theta_mins[j][2] = theta[2] * (minutes[j] / 90);
            theta_mins[j][3] = theta[3] * (minutes[j] / 90) + (90 - minutes[j]) / 90;
    }
}
model {
    vector[3] alpha = rep_vector(1.0, 3); // think if we can do better than this
    theta ~ dirichlet(alpha);
    for (j in 1:nmatch) {
        y[j] ~ multinomial(theta_mins[j]);
    }
}
