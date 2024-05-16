#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int entry(char* symvar) {
    int flag = 0x12345678;
    int var = flag;
    char buf[8];
    strncpy(buf, (char*)(symvar), 9);
    return 0;
}

int main(int ac,char* av[]) {
    char input[50];
    fgets(input, sizeof(input)-1, stdin);
    return entry(input);

}