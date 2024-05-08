#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_ARG_LEN 256


int  entry(const char* s) {
  uint32_t *nullptr = NULL;

  if (strlen(s) == 3) {
    *(nullptr) = 0;
    return 1;
  }
  else {
    return 0;
  }
}

int main(int ac, const char* av[]) {
    char input[MAX_ARG_LEN];

    if (ac != 2)
        return 0;
    return entry(av[1]);
}
