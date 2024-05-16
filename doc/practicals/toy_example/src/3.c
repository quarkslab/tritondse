#include <stdio.h>
#include <inttypes.h>

int entry(char *s) {
    unsigned int symvar = s[0] - 48;
    unsigned int symvar2 = s[1] - 48;
    int arr[30] = {0};
    uint32_t *nullptr = NULL;

    arr[symvar % 30] = s[2];

    if (arr[symvar2 % 30] == 'a')
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
    return entry(input);
}