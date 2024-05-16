#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int entry() {
    char buffer[25];
    int j;
    uint32_t *nullptr = NULL;

    FILE *fp = fopen("tmp.covpro", "r");
    if(fp == NULL) {
        printf("file not found");
        exit(1);
    }
    else {
        fread(buffer, 1, sizeof(buffer), fp);
        sscanf(buffer, "%d", &j);
        fclose(fp);

        if(j == 7) {
            *(nullptr) = 0;
            printf("Crash!\n");
            return 1;
        } else {
            printf("Keep trying!\n");
            return 0;
        }
    }
}

int main(int ac, const char* av[]) {
    return entry();
}