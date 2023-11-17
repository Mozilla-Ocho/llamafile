#ifndef LLAMAFILE_H_
#define LLAMAFILE_H_
#include <stdio.h>
#include <stdbool.h>
#ifdef  __cplusplus
extern "C" {
#endif

struct llamafile;
struct llamafile *llamafile_open(const char *, const char *);
void llamafile_close(struct llamafile *);
long llamafile_read(struct llamafile *, void *, size_t);
long llamafile_write(struct llamafile *, const void *, size_t);
void llamafile_seek(struct llamafile *, size_t, int);
void *llamafile_content(struct llamafile *);
size_t llamafile_tell(struct llamafile *);
size_t llamafile_size(struct llamafile *);
FILE *llamafile_fp(struct llamafile *);

void llamafile_check_cpu(void);
const char *llamafile_get_tmp_dir(void);
bool llamafile_extract(const char *, const char *);
int llamafile_is_file_newer_than(const char *, const char *);
void llamafile_schlep(const void *, size_t);
void llamafile_get_app_dir(char *, size_t);
bool llamafile_launch_browser(const char *);

#ifdef  __cplusplus
}
#endif
#endif /* LLAMAFILE_H_ */
