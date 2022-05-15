#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "test.h"
#include "tensor.h"

int tests_total = 0;
int tests_fail = 0;

double currtime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int within_eps(float a, float b)
{
    return fabs(a-b) < EPS;
}

int same_tensor(tensor a, tensor b)
{
    if(a.n != b.n) {
        fprintf(stderr, "Different dimensionality: %d vs %d\n", a.n, b.n);
        return 0;
    }
    int i;
    for(i = 0; i < a.n; ++i){
        if (a.size[i] != b.size[i]){
            fprintf(stderr, "Dimension %d, different size: %d vs %d\n", i, a.size[i], b.size[i]);
            return 0;
        }
    }
    int len = tensor_len(a);
    for(i = 0; i < len; ++i){
        if (!within_eps(a.data[i], b.data[i])) {
            fprintf(stderr, "Different data at index %d: %f vs %f\n", i, a.data[i], b.data[i]);
            return 0;
        }
    }
    return 1;
}

void test_tensor()
{
    {
        tensor a = tensor_vmake(1, 1);
        tensor b = tensor_vmake(3, 3, 1080, 1920);
        tensor r = tensor_random(1.0f, 3, b.size);

        TEST (a.n == 1);
        TEST (a.size[0] = 1);

        TEST (b.n == 3);
        TEST (b.size[0] == 3);
        TEST (b.size[1] == 1080);
        TEST (b.size[2] == 1920);

        TEST(r.n == 3);
        TEST(r.size[0] == 3);
        TEST(r.size[1] == 1080);
        TEST(r.size[2] == 1920);

        tensor g = tensor_get(r, 1);
        TEST (g.n == 2);
        TEST (g.size[0] == 1080);
        TEST (g.size[1] == 1920);

        tensor h = tensor_get(g, 34);
        TEST (h.n == 1);
        TEST (h.size[0] = 1920);

        tensor i = tensor_get(h, 13);
        TEST (h.n == 1);
        TEST (h.size[0] = 1);

        tensor j = tensor_get(i, 0);

        TEST (same_tensor(i, j));

        tensor_free(a);
        tensor_free(b);
        tensor_free(r);
        tensor_free(g);
        tensor_free(h);
        tensor_free(i);
        tensor_free(j);
    }

    {
        tensor m1 = tensor_vmake(2, 2, 2);
        tensor m2 = tensor_vmake(2, 2, 2);
        m1.data[0] = 1; m1.data[1] = 2;
        m1.data[2] = 3; m1.data[3] = 4;

        m2.data[0] = -1; m2.data[1] = -2;
        m2.data[2] = -3; m2.data[3] = -4;

        tensor m3 = matrix_multiply(m1, m2);
        tensor tm3 = tensor_vmake(2, 2, 2);
        tm3.data[0] = -7; tm3.data[1] = -10;
        tm3.data[2] = -15; tm3.data[3] = -22;

        TEST (same_tensor(m3, tm3));

        tensor_free(m1);
        tensor_free(m2);
        tensor_free(m3);
        tensor_free(tm3);
    }

    {
        int s1[2] = {256, 256};
        int s2[2] = {256, 256};
        tensor r1 = tensor_random(1.0f, 2, s1);
        tensor r2 = tensor_random(1.0f, 2, s2);
        tensor r3 = matrix_multiply(r1, r2);

        TEST (r3.size[0] == 256);
        TEST (r3.size[1] == 256);

        tensor_free(r1);
        tensor_free(r2);
        tensor_free(r3);
    }

    {
        int s[2] = {29, 13};
        tensor t = tensor_random(1.0f, 2, s);
        tensor tt = matrix_transpose(t);
        tensor ttt = matrix_transpose(tt);
        TEST (tt.size[0] == t.size[1]);
        TEST (tt.size[1] == t.size[0]);
        TEST (same_tensor(t, ttt));
        tensor_free(t);
        tensor_free(tt);
        tensor_free(ttt);
    }

    {
        int s[2] = {64, 64};
        tensor t = tensor_random(1.0f, 2, s);
        tensor inv = matrix_invert(t);
        tensor ident = matrix_multiply(t, inv);
        tensor isq = matrix_multiply(ident, ident);
        TEST (same_tensor(ident, isq));
        tensor_free(t);
        tensor_free(inv);
        tensor_free(ident);
        tensor_free(isq);
    }

    {
        /* Testing solving system of equations:
        a + x - 3 y + z = 2
        -5 a + 3 x - 4 y + z = 0
        a + 2 y - z = 1
        a + 2 x = 12
        See: https://www.wolframalpha.com/input/?i=systems+of+equations+calculator&assumption=%22FSelect%22+-%3E+%7B%7B%22SolveSystemOf4EquationsCalculator%22%7D%7D&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation1%22%7D+-%3E%22a+%2B+x+-+3+y+%2B+z+%3D+2%22&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation2%22%7D+-%3E%22-5+a+%2B+3+x+-+4+y+%2B+z+%3D+0%22&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation3%22%7D+-%3E%22a+%2B+2+y+-+z+%3D+1%22&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation4%22%7D+-%3E%22a+%2B+2+x+%3D+12%22
        */

        int s[2] = {4, 4};
        int sb[2] = {4, 1};
        tensor M = tensor_make(2, s);
        tensor b = tensor_make(2, sb);
        M.data[0] = 1; M.data[1] = 1; M.data[2] = -3, M.data[3] = 1;
        M.data[4] = -5; M.data[5] = 3; M.data[6] = -4, M.data[7] = 1;
        M.data[8] = 1; M.data[9] = 0; M.data[10] = 2, M.data[11] = -1;
        M.data[12] = 1; M.data[13] = 2; M.data[14] = 0, M.data[15] = 0;

        b.data[0] = 2; b.data[1] = 0; b.data[2] = 1; b.data[3] = 12;
        tensor a = solve_system(M, b);
        tensor t = tensor_make(2, sb);
        t.data[0] = 22./17; t.data[1] = 91./17; t.data[2] = 84./17; t.data[3] = 173./17;

        TEST (same_tensor(a, t));
    }

    {
        int s[2] = {3, 5};
        tensor t = tensor_random(1.0f, 2, s);
        tensor c = tensor_copy(t);
        TEST (same_tensor(t, c));

        tensor w = tensor_scale(t, 12.3);
        TEST (within_eps(w.data[0], t.data[0]*12.3));
        TEST (within_eps(w.data[11], t.data[11]*12.3));
        tensor_free(t);
        tensor_free(c);
        tensor_free(w);
    }
    {
        tensor t1 = tensor_vmake(4, 11, 5, 13, 9);
        tensor t2 = tensor_vmake(1, 9);
        tensor t3 = tensor_vmake(1, 13);
        tensor t4 = tensor_vmake(2, 1, 9);
        tensor t5 = tensor_vmake(2, 2, 9);
        tensor t6 = tensor_vmake(2, 13, 9);
        tensor t7 = tensor_vmake(4, 11, 1, 13, 1);
        tensor t8 = tensor_vmake(4, 12, 1, 13, 1);
        TEST (tensor_broadcastable(t1, t2) == 1);
        TEST (tensor_broadcastable(t2, t1) == 1);
        TEST (tensor_broadcastable(t1, t3) == 0);
        TEST (tensor_broadcastable(t3, t1) == 0);
        TEST (tensor_broadcastable(t1, t4) == 1);
        TEST (tensor_broadcastable(t4, t1) == 1);
        TEST (tensor_broadcastable(t1, t5) == 0);
        TEST (tensor_broadcastable(t1, t6) == 1);
        TEST (tensor_broadcastable(t1, t7) == 1);
        TEST (tensor_broadcastable(t1, t8) == 0);
        tensor_free(t1);
        tensor_free(t2);
        tensor_free(t3);
        tensor_free(t4);
        tensor_free(t5);
        tensor_free(t6);
        tensor_free(t7);
        tensor_free(t8);
    }
    {
        int s1[3] = {3,2,5};
        int s2[2] = {2,5};
        int s3[1] = {5};
        int s4[2] = {2,1};
        tensor t1 = tensor_random(1, 3, s1);
        tensor t2 = tensor_random(1, 2, s2);
        tensor t3 = tensor_random(1, 1, s3);
        tensor t4 = tensor_random(1, 2, s4);
        tensor o23 = tensor_add(t2, t3);
        tensor o24 = tensor_add(t2, t4);
        //tensor_print(t2);
        //tensor_print(t3);
        //tensor_print(t4);
        //tensor_print(o23);
        //tensor_print(o24);
        tensor t1a = tensor_add(t1, t1);
        tensor t1s = tensor_scale(t1, 2);
        TEST(same_tensor(t1a, t1s));
    }
    {
        int s[2] = {512, 512};
        int i;
        int n = 100;
        tensor a = tensor_random(1, 2, s);
        tensor b = tensor_random(1, 2, s);
        double start = currtime();
        for(i = 0; i < n; ++i){
            tensor c = tensor_add(a, b);
            tensor_free(c);
        }
        double end = currtime();
        printf("tensor_add took %f sec\n", end - start);
        printf("%g sec/op\n", (end-start)/n/(s[0]*s[1]));

        start = currtime();
        for(i = 0; i < n; ++i){
            tensor c = matrix_multiply(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("matrix_multiply took %f sec\n", end - start);
        printf("%g sec/op\n", (end-start)/n/(s[0]*s[1]*s[1]));
    }
}

void test()
{
    test_tensor();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

