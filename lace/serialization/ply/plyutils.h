#include <Python.h>
#include "rply.h"

static PyObject * plyutils_read(PyObject *self, PyObject *args);
static PyObject * plyutils_write(PyObject *self, PyObject *args);
void set_error(const char *message, const char *old_locale);
void error_cb(p_ply ply, const char *message);
int vertex_cb(p_ply_argument argument);
int face_cb(p_ply_argument argument);
const char *backup_locale(void);
void restore_local(const char *old_locale);
