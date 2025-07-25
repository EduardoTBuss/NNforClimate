#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_NODES 8
#define OUTPUT_NODES 2
#define HIDDEN_NODES 16
#define EPOCHS 600
#define LEARNING_RATE 0.001f
#define MAX_DATA 100000
#define SHUFFLE 1

static inline float relu(float x){ 
    return x > 0 ? x : 0; 
}

static inline float drelu(float y){ 
    return y > 0 ? 1 : 0; 
}

static inline float zscore(float x, float mean, float std){ 
    return (x - mean) / (std + 1e-8f); 
}

static inline float inv_zscore(float z, float mean, float std){ 
    return z * std + mean; 
}

static inline float rand_weight(){ 
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; 
}

typedef struct { 
    float input[INPUT_NODES],
    hidden[HIDDEN_NODES],
    output[OUTPUT_NODES];

    float weights_ih[INPUT_NODES][HIDDEN_NODES],
    weights_ho[HIDDEN_NODES][OUTPUT_NODES];

    float bias_h[HIDDEN_NODES], 
    bias_o[OUTPUT_NODES]; 
}NeuralNetwork;

typedef struct { 
    float mean, 
    std; 
} Stats;

static Stats in_stats[INPUT_NODES], out_stats[OUTPUT_NODES];

void init_network(NeuralNetwork *nn) {
    for (int i = 0; i < INPUT_NODES; ++i)
        for (int j = 0; j < HIDDEN_NODES; ++j)
            nn->weights_ih[i][j] = rand_weight();

    for (int i = 0; i < HIDDEN_NODES; ++i) {
        nn->bias_h[i] = rand_weight();

        for (int j = 0; j < OUTPUT_NODES; ++j)
            nn->weights_ho[i][j] = rand_weight();

    }

    for (int i = 0; i < OUTPUT_NODES; ++i)
        nn->bias_o[i] = rand_weight();
}

int load_csv(const char* filename, float x[][INPUT_NODES], float y[][OUTPUT_NODES]) {
    FILE* f = fopen(filename, "r"); 
        if (!f) 
            return 0;

    char line[256]; 
    fgets(line, sizeof(line), f);

    int count = 0;
    while (fgets(line, sizeof(line), f) && count < MAX_DATA) {
        sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &x[count][0], &x[count][1], &x[count][2], &x[count][3],
               &x[count][4], &x[count][5], &x[count][6], &x[count][7],
               &y[count][0], &y[count][1]);
        count++;
    }
    fclose(f); 
    return count;
}

void calc_stats(float x[][INPUT_NODES], float y[][OUTPUT_NODES], int n) {
    for (int j = 0; j < INPUT_NODES; ++j) {
        double sum = 0; 

        for (int i = 0; i < n; ++i) 
            sum += x[i][j];

        in_stats[j].mean = sum / n;

        double var = 0; 
        for (int i = 0; i < n; ++i) 
            var += pow(x[i][j] - in_stats[j].mean, 2);

        in_stats[j].std = sqrt(var / n);
    }

    for (int j = 0; j < OUTPUT_NODES; ++j) {
        double sum = 0; 
        for (int i = 0; i < n; ++i) 
            sum += y[i][j];

        out_stats[j].mean = sum / n;

        double var = 0; 
        for (int i = 0; i < n; ++i) 
            var += pow(y[i][j] - out_stats[j].mean, 2);

        out_stats[j].std = sqrt(var / n);
    }
}

void feedforward(NeuralNetwork *nn) {
    for (int i = 0; i < HIDDEN_NODES; ++i) {
        float sum = nn->bias_h[i];

        for (int j = 0; j < INPUT_NODES; ++j) 
            sum += nn->input[j] * nn->weights_ih[j][i];

        nn->hidden[i] = relu(sum);
    }
    for (int i = 0; i < OUTPUT_NODES; ++i) {
        float sum = nn->bias_o[i];

        for (int j = 0; j < HIDDEN_NODES; ++j) 
            sum += nn->hidden[j] * nn->weights_ho[j][i];

        nn->output[i] = sum;
    }
}

void train(NeuralNetwork *nn, float input[], float target[]) {
    memcpy(nn->input, input, sizeof(float)*INPUT_NODES);
    feedforward(nn);

    float err_o[OUTPUT_NODES];

    for (int i = 0; i < OUTPUT_NODES; ++i) 
        err_o[i] = target[i] - nn->output[i];

    for (int i = 0; i < OUTPUT_NODES; ++i) {
        float grad = err_o[i] * LEARNING_RATE;
        nn->bias_o[i] += grad;

        for (int j = 0; j < HIDDEN_NODES; ++j) 
            nn->weights_ho[j][i] += grad * nn->hidden[j];
    }

    float err_h[HIDDEN_NODES] = {0};

    for (int i = 0; i < HIDDEN_NODES; ++i)
        for (int j = 0; j < OUTPUT_NODES; ++j) 
            err_h[i] += nn->weights_ho[i][j] * err_o[j];

    for (int i = 0; i < HIDDEN_NODES; ++i) {
        float grad = drelu(nn->hidden[i]) * err_h[i] * LEARNING_RATE;

        nn->bias_h[i] += grad;

        for (int j = 0; j < INPUT_NODES; ++j) 
            nn->weights_ih[j][i] += grad * nn->input[j];
    }
}

void shuffle_indices(int *idx, int n) {
    for (int i = n-1; i > 0; --i){
        int j = rand() % (i+1); 
        int tmp = idx[i]; 
        idx[i] = idx[j];
        idx[j] = tmp; 
    }
}

int main() {
    srand((unsigned)time(NULL)); 
    NeuralNetwork nn; 
    init_network(&nn);

    static float x[MAX_DATA][INPUT_NODES], y[MAX_DATA][OUTPUT_NODES];

    int n = load_csv("train.csv", x, y);
        if (!n) { printf("Erro ao ler train.csv\n"); 
            return 1;
        }

    calc_stats(x, y, n);

    static float x_norm[MAX_DATA][INPUT_NODES], y_norm[MAX_DATA][OUTPUT_NODES];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < INPUT_NODES; ++j)
            x_norm[i][j] = zscore(x[i][j], in_stats[j].mean, in_stats[j].std);

        for (int j = 0; j < OUTPUT_NODES; ++j) 
            y_norm[i][j] = zscore(y[i][j], out_stats[j].mean, out_stats[j].std);
    }

    int indices[MAX_DATA]; 
    
    for (int i = 0; i < n; ++i) 
        indices[i] = i;

    for (int e = 0; e < EPOCHS; ++e) {
        if (SHUFFLE) 
            shuffle_indices(indices, n);

        double epoch_loss = 0.0;  
        
        for (int k = 0; k < n; ++k) {
            
            memcpy(nn.input, x_norm[indices[k]], sizeof(nn.input));
            feedforward(&nn);

            float err0 = y_norm[indices[k]][0] - nn.output[0];
            float err1 = y_norm[indices[k]][1] - nn.output[1];
            epoch_loss += err0*err0 + err1*err1;

            train(&nn, x_norm[indices[k]], y_norm[indices[k]]);
        }

        double mse = epoch_loss / (n * OUTPUT_NODES);
        printf("Epoch %3d/%d  -  Loss (MSE): %.9f\n", e+1, EPOCHS, mse);
    }


    printf("Treinamento finalizado!\n");

    while(1){
        const char* prompts[INPUT_NODES] = {"Temperatura", "Umidade", "Vento", "Hora", "Dia", "Nuvens", "Pressao", "Precipitacao"};
        float user_in[INPUT_NODES];

        for (int j = 0; j < INPUT_NODES; ++j) {
            printf("%s: ", prompts[j]);
            scanf("%f", &user_in[j]);
        }

        float norm_in[INPUT_NODES];

        for (int j = 0; j < INPUT_NODES; ++j) 
            norm_in[j] = zscore(user_in[j], in_stats[j].mean, in_stats[j].std);

        memcpy(nn.input, norm_in, sizeof(norm_in));
        feedforward(&nn);

        float out0 = inv_zscore(nn.output[0], out_stats[0].mean, out_stats[0].std);
        float out1 = inv_zscore(nn.output[1], out_stats[1].mean, out_stats[1].std);

        printf("Sensacao: %.2fC, Probabilidade de Chuva: %.2f%%\n", out0, out1);
    }
}