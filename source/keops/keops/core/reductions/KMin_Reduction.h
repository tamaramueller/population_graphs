#pragma once

#include <sstream>

#include "core/reductions/Reduction.h"
#include "core/reductions/Zero_Reduction.h"
#include "core/pre_headers.h"
#include "core/utils/Infinity.h"
#include "core/utils/TypesUtils.h"

namespace keops {
// Implements the k-min-arg-k-min reduction operation : for each i or each j, find the values and indices of the
// k minimal values of Fij
// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.

template < class F, int K, int tagI=0 >
struct KMin_ArgKMin_Reduction : public Reduction<F,tagI> {

    static const int DIM = 2*K*F::DIM;		// DIM is dimension of output of convolution ; for a arg-k-min reduction it is equal to the dimension of output of formula

    static const int DIMRED = DIM;	// dimension of temporary variable for reduction
		
    static void PrintId(::std::stringstream& str) {
        str << "KMin_ArgKMin_Reduction(";			// prints "("
        F::PrintId(str);				// prints the formula F
        str << ",K=" << K << ",tagI=" << tagI << ")";
    }

    template < typename TYPEACC, typename TYPE >
    struct InitializeReduction {
        DEVICE INLINE void operator()(TYPEACC *tmp) {
            #pragma unroll
            for(int k=0; k<F::DIM; k++) {
                #pragma unroll
                for(int l=k; l<K*2*F::DIM+k; l+=2*F::DIM) {
                    tmp[l] = cast_to<TYPEACC>(PLUS_INFINITY<TYPE>::value); // initialize output
                    tmp[l+F::DIM] = cast_to<TYPEACC>(0.0f); // initialize output
                }
            }
#if USE_HALF && GPU_ON
            // to be continued...
            if (threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) {
              printf("\n   [KeOps] Error : KMin or ArgKMin reductions are not yet implemented with half precision type.\n\n");
              asm("trap;");
            }
#endif
        }
    };


    // equivalent of the += operation
    template < typename TYPEACC, typename TYPE >
    struct ReducePairShort {
        DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi, TYPE val) {
#if !USE_HALF
            TYPE xik;
            int l;
            #pragma unroll
            for(int k=0; k<F::DIM; k++) {
                xik = xi[k];
                #pragma unroll
                for(l=(K-1)*2*F::DIM+k; l>=k && xik<tmp[l]; l-=2*F::DIM) {
                    TYPE tmpl = tmp[l];
                    int indtmpl = tmp[l+F::DIM];
                    tmp[l] = xik;
                    tmp[l+F::DIM] = val;
                    if(l<(K-1)*2*F::DIM+k) {
                        tmp[l+2*F::DIM] = tmpl;
                        tmp[l+2*F::DIM+F::DIM] = indtmpl;
                    }
                }
            }
#endif
        }
    };

	// equivalent of the += operation
	template < typename TYPEACC, typename TYPE >
	struct ReducePair {
		DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi) {
#if !USE_HALF
            TYPE out[DIMRED];
            
            #pragma unroll
			for(int k=0; k<F::DIM; k++) {
			    int p = k;
			    int q = k;
                            #pragma unroll
			    for(int l=k; l<DIMRED; l+=2*F::DIM) {
			        if(xi[p]<tmp[q]) {
					    out[l] = xi[p];
					    out[F::DIM+l] = xi[F::DIM+p];
					    p += 2*F::DIM;
					}
					else {
					    out[l] = tmp[q];
					    out[F::DIM+l] = tmp[F::DIM+q];
					    q += 2*F::DIM;
					}  
				}
			}
                        #pragma unroll
			for(int k=0; k<DIMRED; k++)
			    tmp[k] = out[k];
#endif
		}
	};
        
    template < typename TYPEACC, typename TYPE >
    struct FinalizeOutput {
        DEVICE INLINE void operator()(TYPE *acc, TYPE *out, int i) {
            #pragma unroll
            for(int k=0; k<DIM; k++)
                out[k] = acc[k];
        }
    };

    // no gradient implemented here

};

// Implements the arg-k-min reduction operation : for each i or each j, find the indices of the
// k minimal values of Fij
// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.

template < class F, int K, int tagI=0 >
struct ArgKMin_Reduction : public KMin_ArgKMin_Reduction<F,K,tagI> {


    static const int DIM = K*F::DIM;		// DIM is dimension of output of convolution ; for a arg-k-min reduction it is equal to the dimension of output of formula

    static void PrintId(::std::stringstream& str) {
        str << "ArgKMin_Reduction(";			// prints "("
        F::PrintId(str);				// prints the formula F
        str << ",K=" << K << ",tagI=" << tagI << ")";
    }
                  
    template < typename TYPEACC, typename TYPE >
    struct FinalizeOutput {
        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, int i) {
            #pragma unroll
            for(int k=0; k<F::DIM; k++)
                #pragma unroll
                for(int p=k, l=k; l<K*2*F::DIM+k; p+=F::DIM, l+=2*F::DIM)
                    out[p] = acc[l+F::DIM];
        }
    };

    template < class V, class GRADIN >
    using DiffT = Zero_Reduction<V::DIM,(V::CAT)%2>;
    // remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j.
    // In this case there is a summation left to be done by the user.


};

// Implements the k-min reduction operation : for each i or each j, find the
// k minimal values of Fij
// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.

template < class F, int K, int tagI=0 >
struct KMin_Reduction : public KMin_ArgKMin_Reduction<F,K,tagI> {


        static const int DIM = K*F::DIM;		// DIM is dimension of output of convolution ; for a arg-k-min reduction it is equal to the dimension of output of formula
                 
    static void PrintId(::std::stringstream& str) {
        str << "KMin_Reduction(";			// prints "("
        F::PrintId(str);				// prints the formula F
        str << ",K=" << K << ",tagI=" << tagI << ")";
    }

    template < typename TYPEACC, typename TYPE >
    struct FinalizeOutput {
        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, int i) {
            #pragma unroll
            for(int k=0; k<F::DIM; k++)
                #pragma unroll
                for(int p=k, l=k; l<K*2*F::DIM+k; p+=F::DIM, l+=2*F::DIM)
                    out[p] = acc[l];
        }
    };

    // no gradient implemented here


};

#define KMin_ArgKMin_Reduction(F,K,I) KeopsNS<KMin_ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define ArgKMin_Reduction(F,K,I) KeopsNS<ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define KMin_Reduction(F,K,I) KeopsNS<KMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()


}