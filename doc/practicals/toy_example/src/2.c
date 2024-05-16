#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int entry(char* s) {
    unsigned char symvar = s[0] - 48;
    int ary[] = {1,2,3,4,5};
    uint32_t *nullptr = NULL;

    if(ary[symvar%5] == 5){
        *(nullptr) = 0;
        return 1;
    }
    else {
        return 0;
    }
}

int main(int ac, char* av[]) {
    return entry(av[1]);
}