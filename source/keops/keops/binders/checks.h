#pragma once

#include <string>
#include <tuple>
#include <functional>
#include <algorithm>

#include "binders/utils.h"
#include "binders/keops_cst.h"


namespace keops_binders {

/////////////////////////////////////////////////////////////////////////////////
//                    Sanity checks on args
/////////////////////////////////////////////////////////////////////////////////

void check_nargs(int narg, int keops_nminargs) {
  if (narg < keops_nminargs)
    keops_error("[KeOps] Wrong number of args : is " + std::to_string(narg)
                + " but should be at least " + std::to_string(keops_nminargs)
                + " in formula.");
}

void check_tag(int tag, std::string msg) {
  if ((tag < 0) || (tag > 1)) {
    keops_error("[KeOps] tag" + msg + " should be (0 or 1) but is " + std::to_string(tag));
  }
}

template< typename array_t >
void check_contiguity(array_t &obj_ptr, int i) {
  if (!is_contiguous(obj_ptr)) {
    keops_error("[Keops] Arg at position " + std::to_string(i) + ": is not contiguous. "
                + "Please provide 'contiguous' dara array, as KeOps does not support strides. "
                + "If you're getting this error in the 'backward' pass of a code using torch.sum() "
                + "on the output of a KeOps routine, you should consider replacing 'a.sum()' with "
                + "'torch.dot(a.view(-1), torch.ones_like(a).view(-1))'. ");
  }
}


template< typename array_t >
// This class contains get the sizes of the input args of a formula.
class Sizes {
public:
  // constructors
  Sizes(int _nargs, array_t* args, int _nx, int _ny) {
	  
    nargs = _nargs;
	nx = _nx;
	ny = _ny;

    // fill shapes wit "batch dimensions" [A, .., B], the table will look like:
    //
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    fill_shape(_nargs, args);

    check_ranges(_nargs, args);

    shapes = &_shapes[0];
   
    // fill shape_out
    _shape_out.resize(nbatchdims + 3);
    #if C_CONTIGUOUS
    std::copy(_shapes.begin(), _shapes.begin() + nbatchdims + 3, _shape_out.begin());// Copy the "batch dimensions"
    _shape_out.erase(_shape_out.begin() + nbatchdims + (1 - keops_tagIJ));
    #else
    std::reverse_copy(_shapes.begin(), _shapes.begin() + nbatchdims + 3,
                      _shape_out.begin());// Copy the "batch dimensions"
    _shape_out.erase(_shape_out.begin() + 1 + keops_tagIJ);
    #endif

    shape_out = &_shape_out[0];

    // fill nx and ny
    M = _shapes[nbatchdims];      // = M
    N = _shapes[nbatchdims + 1];  // = N

    // Compute the product of all "batch dimensions"
    nbatches = 1;
    for (int b = 0; b < nbatchdims; b++)
      nbatches *= shapes[b];
    //int nbatches = std::accumulate(shapes, shapes + nbatchdims, 1, std::multiplies< int >());
    
    nx = nbatches * M;  // = A * ... * B * M
    ny = nbatches * N;  // = A * ... * B * N

  }
  
  // attributs
  int nargs;
  int nx, ny;
  int M, N;
  int nbatchdims;
  int nbatches;
  
  std::vector< int > _shapes;
  int* shapes;
  std::vector< int > _shape_out;
  int* shape_out;
  
  // methods

  void switch_to_half2_indexing();

private:
  void fill_shape(int nargs, array_t* args);
  
  void check_ranges(int nargs, array_t* args);
  
  std::function< int(array_t, int, int) > get_size_batch;
  int MN_pos, D_pos;
};


template< typename array_t >
void Sizes< array_t >::fill_shape(int nargs, array_t* args) {

  int pos = std::max(keops_pos_first_argI, keops_pos_first_argJ);

  if (pos > -1) {
    // Are we working in batch mode? Infer the answer from the first arg =============
    nbatchdims =  get_ndim(args[pos]) - 2;  // number of batch dimensions = Number of dims of the first tensor - 2

    if (nbatchdims < 0) {
      keops_error("[KeOps] Wrong number of dimensions for arg at position 0: is "
                  + std::to_string(get_ndim(args[0])) + " but should be at least 2."
      );
    }
  } else {
    nbatchdims = 0;
  }

  #if C_CONTIGUOUS
  get_size_batch = [](auto args, int nbatch, int b) {
    return get_size(args, b);
  };
  MN_pos = nbatchdims;
  D_pos = nbatchdims + 1;
  #else
  D_pos = 0;
  MN_pos = 1;
  get_size_batch = [](auto obj_ptr, int nbatch, int b) {
    return get_size(obj_ptr, nbatch - b);
  };
  #endif

  // Now, we'll keep track of the output + all arguments' shapes in a large array:
  _shapes.resize((nargs + 1) * (nbatchdims + 3), 1);
  
  #if USE_HALF
    if(keops_tagIJ==0) {
        _shapes[nbatchdims] = nx%2? nx+1 : nx;
        _shapes[nbatchdims + 1] = 2*ny;
    } else {
        _shapes[nbatchdims] = 2*nx;
        _shapes[nbatchdims + 1] = ny%2? ny+1 : ny;
    }
  #else
    _shapes[nbatchdims] = nx;
    _shapes[nbatchdims + 1] = ny;
  #endif
  
  _shapes[nbatchdims + 2] = keops_dimout;   // Top right corner: dimension of the output
  
}


template< typename array_t >
void Sizes< array_t >::check_ranges(int nargs, array_t* args) {
//void check_ranges(int nargs, array_t* args) {
	
  // Check the compatibility of all tensor shapes ==================================
  if (keops_nminargs > 0) {
    
    // Checks args in all the positions that correspond to "i" variables:
    for (int k = 0; k < keops_nvarsI; k++) {
      int i = keops_indsI[k];
      // Fill in the (i+1)-th line of the "shapes" array ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);
      
      // Check the number of dimensions --------------------------------------------
      int ndims = get_ndim(args[i]);  // Number of dims of the i-th tensor
      
      if (ndims != nbatchdims + 2) {
        keops_error("[KeOps] Wrong number of dimensions for arg at position " + std::to_string(i)
                    + " (i type): KeOps detected " + std::to_string(nbatchdims)
                    + " batch dimensions from the first argument 0, and thus expected "
                    + std::to_string(nbatchdims + 2)
                    + " dimensions here, but only received "
                    + std::to_string(ndims)
                    + ". Note that KeOps supports broadcasting on batch dimensions, "
                    + "but still expects 'dummy' unit dimensions in the input shapes, "
                    + "for the sake of clarity.");
      }
  
  
  
      // First, the batch dimensions:
      for (int b = 0; b < nbatchdims; b++) {
        _shapes[off_i + b] = get_size_batch(args[i], nbatchdims + 2, b);
        
        // Check that the current value is compatible with what
        // we've encountered so far, as stored in the first line of "shapes"
        if (_shapes[off_i + b] != 1) {  // This dimension is not "broadcasted"
          if (_shapes[b] == 1) {
            _shapes[b] = _shapes[off_i + b];  // -> it becomes the new standard
          } else if (_shapes[b] != _shapes[off_i + b]) {
            keops_error("[KeOps] Wrong value of the batch dimension "
                        + std::to_string(b) + " for argument number " + std::to_string(i)
                        + " : is " + std::to_string(_shapes[off_i + b])
                        + " but was " + std::to_string(_shapes[b])
                        + " or 1 in previous arguments.");
          }
        }
      }
  
      _shapes[off_i + nbatchdims] = get_size(args[i], MN_pos);  // = "M"
      _shapes[off_i + nbatchdims + 2] = get_size(args[i], D_pos);  // = "D"
      
      // Check the number of "lines":
      if (_shapes[nbatchdims] != _shapes[off_i + nbatchdims]) {
        keops_error("[KeOps] Wrong value of the 'i' dimension "
                    + std::to_string(nbatchdims) + "for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(_shapes[off_i + nbatchdims])
                    + " but was " + std::to_string(_shapes[nbatchdims])
                    + " in previous 'i' arguments.");
      }
      
      // And the number of "columns":
      if (_shapes[off_i + nbatchdims + 2] != static_cast< int >(keops_dimsX[k])) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                    + std::to_string(nbatchdims + 1) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(_shapes[off_i + nbatchdims + 2])
                    + " but should be " + std::to_string(keops_dimsX[k]));
      }
  
      check_contiguity(args[i], i);
    }
      
    // Checks args in all the positions that correspond to "j" variables:
    for (int k = 0; k < keops_nvarsJ; k++) {
      int i = keops_indsJ[k];
      
      // Check the number of dimensions --------------------------------------------
      int ndims = get_ndim(args[i]);  // Number of dims of the i-th tensor
      
      if (ndims != nbatchdims + 2) {
        keops_error("[KeOps] Wrong number of dimensions for arg at position " + std::to_string(i)
                    + " (j type): KeOps detected " + std::to_string(nbatchdims)
                    + " batch dimensions from the first argument 0, and thus expected "
                    + std::to_string(nbatchdims + 2)
                    + " dimensions here, but only received "
                    + std::to_string(ndims)
                    + ". Note that KeOps supports broadcasting on batch dimensions, "
                    + "but still expects 'dummy' unit dimensions in the input shapes, "
                    + "for the sake of clarity.");
      }
      
      // Fill in the (i+1)-th line of the "shapes" array ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);
      
      // First, the batch dimensions:
      for (int b = 0; b < nbatchdims; b++) {
        _shapes[off_i + b] = get_size_batch(args[i], nbatchdims + 2, b);
        
        // Check that the current value is compatible with what
        // we've encountered so far, as stored in the first line of "shapes"
        if (_shapes[off_i + b] != 1) {  // This dimension is not "broadcasted"
          if (_shapes[b] == 1) {
            _shapes[b] = _shapes[off_i + b];  // -> it becomes the new standard
          } else if (_shapes[b] != _shapes[off_i + b]) {
            keops_error("[KeOps] Wrong value of the batch dimension "
                        + std::to_string(b) + " for argument number " + std::to_string(i)
                        + " : is " + std::to_string(_shapes[off_i + b])
                        + " but was " + std::to_string(_shapes[b])
                        + " or 1 in previous arguments.");
          }
        }
      }
 
      _shapes[off_i + nbatchdims + 1] = get_size(args[i], MN_pos);  // = "N"
      _shapes[off_i + nbatchdims + 2] = get_size(args[i], D_pos);  // = "D"
      
      // Check the number of "lines":
      if (_shapes[nbatchdims + 1] != _shapes[off_i + nbatchdims + 1]) {
        keops_error("[KeOps] Wrong value of the 'j' dimension "
                    + std::to_string(nbatchdims) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(shapes[off_i + nbatchdims + 1])
                    + " but was " + std::to_string(shapes[nbatchdims + 1])
                    + " in previous 'j' arguments.");
      }
      
      // And the number of "columns":
      if (_shapes[off_i + nbatchdims + 2] != static_cast< int >(keops_dimsY[k])) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                    + std::to_string(nbatchdims + 1) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(_shapes[off_i + nbatchdims + 2])
                    + " but should be " + std::to_string(keops_dimsY[k]));
      }
  
      check_contiguity(args[i], i);
    }
    
    for (int k = 0; k < keops_nvarsP; k++) {
      int i = keops_indsP[k];
      // Fill in the (i+1)-th line of the "shapes" array ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);
      // First, the batch dimensions:
      for (int b = 0; b < nbatchdims; b++) {
        _shapes[off_i + b] = get_size_batch(args[i], nbatchdims + 2, b);
      }
      _shapes[off_i + nbatchdims + 2] = get_size(args[i], nbatchdims);  // = "D" 
#if USE_HALF  
      int dim_param = _shapes[off_i + nbatchdims + 2] / 2;
#else
      int dim_param = _shapes[off_i + nbatchdims + 2];
#endif
      if (dim_param != static_cast< int >(keops_dimsP[k])) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                    + std::to_string(nbatchdims) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(dim_param)
                    + " but should be " + std::to_string(keops_dimsP[k]));
      }
      check_contiguity(args[i], i);
    }
  }
  
  
}

template< typename array_t >
void Sizes< array_t >::switch_to_half2_indexing() {
    // special case of float16 inputs : because we use half2 type in Cuda codes, we need to divide by two nx, ny, and M, N, or D
    // values inside the shapes vector.
    nx = nx / 2;
    ny = ny / 2;
    M = M / 2;
    N = N / 2;
    shapes[nbatchdims] = shapes[nbatchdims] / 2;
    shapes[nbatchdims+1] = shapes[nbatchdims+1] / 2;
    for (int i = 0; i < nargs; i++) {
        int off_i = (i + 1) * (nbatchdims + 3);
        // we don't have anymore the category information...
        // the last three dimensions are either of the form (M,1,D), (1,N,D), or (1,1,D)
        // where M or N are even in the 2 first cases, or D is even in the third case.
        if(shapes[off_i+nbatchdims]>1)
            shapes[off_i+nbatchdims] = shapes[off_i+nbatchdims] / 2;
        else if(shapes[off_i+nbatchdims+1]>1)
            shapes[off_i+nbatchdims+1] = shapes[off_i+nbatchdims+1] / 2;
        else
            shapes[off_i+nbatchdims+2] = shapes[off_i+nbatchdims+2] / 2;   
    }
}

}
