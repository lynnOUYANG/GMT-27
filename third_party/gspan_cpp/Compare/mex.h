#ifndef GQT_GSPAN_CPP_MEX_STUB_H
#define GQT_GSPAN_CPP_MEX_STUB_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>

typedef uint32_t uint32_T;
typedef struct mxArray_tag mxArray;

enum mxClassID {
    mxDOUBLE_CLASS = 0,
    mxUINT32_CLASS = 1
};

inline void mexPrintf(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

inline void mexErrMsgTxt(const char *msg)
{
    std::fprintf(stderr, "%s\n", msg);
    std::abort();
}

inline unsigned int mxGetN(const mxArray *) { return 0; }
inline unsigned int mxGetM(const mxArray *) { return 0; }
inline void *mxGetPr(const mxArray *) { return nullptr; }
inline mxArray *mxGetField(const mxArray *, int, const char *) { return nullptr; }
inline int mxIsUint32(const mxArray *) { return 0; }
inline int mxIsCell(const mxArray *) { return 0; }
inline double mxGetScalar(const mxArray *) { return 0.0; }
inline mxArray *mxCreateStructMatrix(int, int, int, const char **) { return nullptr; }
inline mxArray *mxCreateNumericMatrix(int, int, mxClassID, int) { return nullptr; }
inline mxArray *mxCreateCellMatrix(int, int) { return nullptr; }
inline void mxSetField(mxArray *, int, const char *, mxArray *) {}
inline void mxSetCell(mxArray *, int, mxArray *) {}

#endif
