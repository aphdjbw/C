#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int label;
    double score;
} Prediction;

int cmp(const void *a, const void *b) {
    return ((Prediction*)b)->score > ((Prediction*)a)->score ? 1 : -1;
}

void calcPRF(int *pred, int *true, int n, double th, double *P, double *R, double *F1) {
    int TP=0, FP=0, FN=0;
    for(int i=0; i<n; i++) {
        int p = pred[i]>=th ? 1 : 0;
        TP += (p&&true[i]);
        FP += (p&&!true[i]);
        FN += (!p&&true[i]);
    }
    *P = (TP+FP) ? (double)TP/(TP+FP) : 0;
    *R = (TP+FN) ? (double)TP/(TP+FN) : 0;
    *F1 = (*P+*R) ? 2**P**R/(*P+*R) : 0;
}

double calcAUC(Prediction *ps, int n) {
    qsort(ps, n, sizeof(Prediction), cmp);
    int pos=0, neg=0;
    double sum=0;
    for(int i=0; i<n; i++) {
        if(ps[i].label) { sum += i+1; pos++; }
        else neg++;
    }
    return (pos&&neg) ? (sum - pos*(pos+1)/2.0)/(pos*neg) : 0;
}

int main() {
    int true[] = {1,0,1,1,0}, pred[] = {85,70,92,60,45};
    int n = sizeof(true)/sizeof(true[0]);
    double P,R,F1,th=75;

    calcPRF(pred, true, n, th, &P, &R, &F1);
    printf("P:%.4f R:%.4f F1:%.4f\n", P, R, F1);

    Prediction ps[n];
    for(int i=0; i<n; i++) { ps[i].label=true[i]; ps[i].score=pred[i]/100.0; }
    printf("AUC:%.4f\n", calcAUC(ps, n));
    return 0;
}
# C