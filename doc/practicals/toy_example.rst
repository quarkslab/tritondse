Toy Example
===========

These toy examples will present you various use-cases containing a bug to trigger.
The goal is to trigger them using the tritondse exploration.


1. Non standard input
---------------------

By default tritondse is able to inject input either on stdin either on argv.
The goal here is to trigger the bug by injection the symbolisation at an arbitrary
location.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>

    int entry() {
        char buffer[25];
        int j;

        FILE *fp = fopen("tmp.covpro", "r");
        if(fp == NULL) {
            printf("file not found");
            exit(1);
        }
        else {
            fread(buffer, 1, sizeof(buffer), fp);
            fscanf(buffer,"%d",&j);
            fclose(fp);

            if(j == 7) {
                *(nullptr) = 0;
                return 1;
            } else {
                return 0;
            }
        }
    }

    int main(int ac, const char* av[]) {
        return entry();
    }


2. Symbolic read
----------------

By default the exploration just negate branches, but does not try to perform
state coverage on pointer values *(as it raises a lot of test-cases potentially
not interesting)*. The goal here is to perform manual state coverage on pointer
values.

.. code-block:: c

    int entry(char* s) {

        unsigned char symvar = s[0] - 48;
        int ary[] = {1,2,3,4,5};

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


3. Symbolic read & write
------------------------

Same principle here, except that triggering the bug require resolving some
kind of a pointer aliasing issue.

.. code-block:: c

    int entry(char *s) {
        unsigned int symvar = s[0] - 48;
        unsigned int symvar2 = s[1] - 48;
        int arr[30] = {0};

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


4. String length
----------------

Symbolic execution hardly infers 'meta-properties' of data. For string its length
is a meta-property that the symbolic executor does not know how to mutate. It can
be an issue when performing coverage.

.. code-block:: c

    #include <string.h>

    int  entry(const char* s) {

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


5. Off-by-One example
---------------------

Write a simple intrinsic function to obtain the stack buffer size
during exploration, and write a simple sanitizer for `strncpy` that
checks that no buffer overflow is taking place.

.. code-block:: c

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

