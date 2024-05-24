#ifndef LLAMAFILE_H_
#define LLAMAFILE_H_
#include <stdbool.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

struct llamafile;
struct llamafile *llamafile_open_gguf(const char *, const char *);
void llamafile_close(struct llamafile *);
long llamafile_read(struct llamafile *, void *, size_t);
long llamafile_write(struct llamafile *, const void *, size_t);
bool llamafile_seek(struct llamafile *, size_t, int);
void *llamafile_content(struct llamafile *);
size_t llamafile_tell(struct llamafile *);
size_t llamafile_size(struct llamafile *);
FILE *llamafile_fp(struct llamafile *);

void llamafile_govern(void);
void llamafile_check_cpu(void);
void llamafile_help(const char *);
void llamafile_log_command(char *[]);
const char *llamafile_get_tmp_dir(void);
bool llamafile_has(char **, const char *);
bool llamafile_extract(const char *, const char *);
int llamafile_is_file_newer_than(const char *, const char *);
void llamafile_schlep(const void *, size_t);
void llamafile_get_app_dir(char *, size_t);
void llamafile_launch_browser(const char *);

extern bool FLAG_trap;
extern bool FLAG_precise;
extern bool FLAG_unsecure;

#define LLAMAFILE_GPU_ERROR -2
#define LLAMAFILE_GPU_DISABLE -1
#define LLAMAFILE_GPU_AUTO 0
#define LLAMAFILE_GPU_AMD 1
#define LLAMAFILE_GPU_APPLE 2
#define LLAMAFILE_GPU_NVIDIA 4
extern int FLAG_gpu;
extern bool FLAG_tinyblas;
extern bool FLAG_nocompile;
extern bool FLAG_recompile;
bool llamafile_has_gpu(void);
int llamafile_gpu_layers(int);
bool llamafile_has_cuda(void);
bool llamafile_has_metal(void);
bool llamafile_has_amd_gpu(void);
int llamafile_gpu_parse(const char *);
const char *llamafile_describe_gpu(void);

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_H_ */
