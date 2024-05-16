#include <stdlib.h>
#include <stdio.h>

int entry(char *s1, char* s2) {

    int *nullptr = NULL;
    if (*s1 == 'a' && *s2 == 'b')
    {
        *(nullptr) = 0;
        return 1;
    }
    else
        return 0;
}

int main(int ac, char *av[]) {
    char input[25];

    fgets(input, sizeof(input) - 1, stdin);

    if (ac != 2)
        return 1;
    return entry(input, av[1]);
}