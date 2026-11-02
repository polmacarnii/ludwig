#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

#define EPS 1e-5 /* keeps log() away from zero */
#define INV_32768 (1.0 / 32768.0)
#define INV_WIN (1.0 / WIN)

double hann[WIN];
double tw_re[WIN / 2 + 1];
double tw_im[WIN / 2 + 1];
int rev[WIN];

void hang(const char *msg, const char *path) {
  if (path) {
    fprintf(stderr, "lsd: %s: %s\n", path, msg);
  } else {
    fprintf(stderr, "lsd: %s\n", msg);
  }

  exit(EXIT_FAILURE);
}

/* Our .raw is at int16 samples so thats why the function is int16_t */
int16_t *read_raw(const char *path, size_t *out_n) {
  FILE *f;
  long size;
  size_t nbytes, n, i;
  unsigned char *bytes;
  int16_t *samples;

  f = fopen(path, "rb");

  if (!f) {
    hang(strerror(errno), path);
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    hang("cannot seek", path);
  }

  size = ftell(f);

  if (size < 0) {
    hang("cannot look size", path);
  }

  rewind(f);

  /* a trailing odd byte is half a sample and its ignored */
  nbytes = ((size_t)size / 2) * 2;
  n = nbytes / 2;

  if (n < WIN) {
    hang("too short (its less than 2048 samples)", path);
  }

  bytes = malloc(nbytes);
  samples = malloc(n * sizeof *samples);

  if (!bytes || !samples) {
    hang("Out of memory", NULL);
  }

  if (fread(bytes, 1, nbytes, f) != nbytes) {
    hang("short read", path);
  }

  fclose(f);

  for (i = 0; i < n; i++) {
    samples[i] = (int16_t)(bytes[2 * i] | (bytes[2 * i + 1] << 8));
  }
  free(bytes);
  *out_n = n;
  return samples;
}

/* Hann window, FFT bit-reversal table and the twiddle factors */
void setup() {
  int i, j, bits = 0;

  for (i = 1; i < WIN; i <<= 1) {
    bits++;
  }

  for (i = 0; i < WIN; i++) {
    int r = 0;

    for (j = 0; j < bits; j++) {
      r = (r << 1) | ((i >> j) & 1);
    }

    rev[i] = r;
  }

  for (i = 0; i <= WIN / 2; i++) {
    double a = -2.0 * M_PI * i / WIN;
    tw_re[i] = cos(a);
    tw_im[i] = sin(a);
  }

  for (i = 0; i < WIN; i++) {
    hann[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / WIN));
  }
}

void fft(double *re, double *im) {
  int i, j, len;

  for (i = 0; i < WIN; i++) {
    j = rev[i];
    if (i < j) {
      double t;
      t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }

  for (len = 2; len <= WIN; len <<= 1) {
    int half = len >> 1;
    int step = WIN / len;

    for (i = 0; i < WIN; i += len) {
      for (j = 0; j < half; j++) {
        int k = j * step;
        double wr = tw_re[k], wi = tw_im[k];
        double ur = re[i + j], ui = im[i + j];
        double vr = re[i + j + half] * wr - im[i + j + half] * wi;
        double vi = re[i + j + half] * wi + im[i + j + half] * wr;

        re[i + j] = ur + vr;
        im[i + j] = ui + vi;
        re[i + j + half] = ur - vr;
        im[i + j + half] = ui - vi;
      }
    }
  }
}

void fill_window(const int16_t *src, double *re, double *im) {
  int i;

  for (i = 0; i < WIN; i++) {
    re[i] = (double)src[i] * INV_32768 * hann[i];
    im[i] = 0.0;
  }
}

double logmag(double re, double im) {
  return log(sqrt(re * re + im * im) * INV_WIN + EPS);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s reference.raw candidate.raw\n", argv[0]);
    return EXIT_FAILURE;
  }

  int16_t *ref, *cand;
  size_t nref, ncand, n, nframes, f;
  double re[WIN], im[WIN], logref[WIN / 2 + 1];
  double total = 0.0;
  int first, last, b, nbands;

  setup();

  ref = read_raw(argv[1], &nref);
  cand = read_raw(argv[2], &ncand);

  /* if the length differ then just compare only what both have */
  if (nref < ncand) {
    n = nref;
  } else {
    n = ncand;
  }

  nframes = 1 + (n - WIN) / HOP;

  first = (int)ceil((double)CUTOFF_HZ * WIN / SAMPLERATE);
  last = WIN / 2;

  if (first > last) {
    hang("cutoff leaves no bands to compare", NULL);
  }

  nbands = last - first + 1;

  for (f = 0; f < nframes; f++) {
    size_t start = f * HOP;
    double acc = 0.0;

    fill_window(ref + start, re, im);
    fft(re, im);
    for (b = first; b <= last; b++) {
      logref[b] = logmag(re[b], im[b]);
    }

    fill_window(cand + start, re, im);
    fft(re, im);
    for (b = first; b <= last; b++) {
      double d = logref[b] - logmag(re[b], im[b]);
      acc += d * d;
    }

    total += sqrt(acc / nbands);
  }

  printf("%.6f\n", total / nframes);

  free(ref);
  free(cand);
  return 0;
}
