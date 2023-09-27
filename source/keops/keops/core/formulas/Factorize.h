#pragma once

#include <sstream>

#include "core/pack/UnivPack.h"
#include "core/pack/Pack.h"
#include "core/pack/ConcatPack.h"
#include "core/pack/GetInds.h"
#include "core/autodiff/BinaryOp.h"
#include "core/autodiff/Var.h"
#include "core/autodiff/CountIn.h"
#include "core/formulas/constants/IntConst.h"
#include "core/pre_headers.h"


//////////////////////////////////////////////////////////////
////      FACTORIZE OPERATOR  : Factorize< F,G >          ////
//////////////////////////////////////////////////////////////

// Factorize< F,G > is the same as F, but when evaluating we factorize
// the computation of G, meaning that if G appears several times inside the
// formula F, we will compute it once only
namespace keops {

template < class F, class G >
struct Factorize_Alias;
template < class F, class G >
using Factorize = typename Factorize_Alias< F, G >::type;
template < class F, class G >
using CondFactorize = CondType< Factorize< F, G >, F, (CountIn< F, G >::val > 1) >;

template < class F, class G >
struct Factorize_Impl : BinaryOp< Factorize_Impl, F, G > {

  static const int DIM = F::DIM;

  static void PrintId(::std::stringstream &str) {
    using IndsTempVars = GetInds< typename F::template VARS< 3>>;
    static const int dummyPos = 1 + IndsTempVars::MAX;
    using dummyVar = Var< dummyPos, G::DIM, 3 >;
    using Ffact = typename F::template Replace< G, dummyVar >;
    str << "[";
    dummyVar::PrintId(str);
    str << "=";
    G::PrintId(str);
    str << ";";
    Ffact::PrintId(str);
    str << "]";
  }

  using THIS = Factorize_Impl< F, G >;

  using Factor = G;

  // we define a new formula from F (called factorized formula), replacing G inside by a new variable ; this is used in function Eval()
  template < class INDS >
  using FactorizedFormula = typename F::template Replace< G,
                                                          Var< INDS::MAX + 1,
                                                               G::DIM,
                                                               3>>;    // means replace G by Var<INDS::SIZE,G::DIM,3> in formula F

  template < class INDS, typename TYPE, typename ...ARGS >
  static DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
    // First we compute G
    TYPE outG[G::DIM];
    G::template Eval< INDS >(outG, args...);
    // Ffact is the factorized formula
    using Ffact = typename THIS::template FactorizedFormula< INDS >;
    // new indices for the call to Eval : we add one more index to the list
    using NEWINDS = ConcatPacks< INDS, pack< INDS::MAX + 1>>;
    // call to Eval on the factorized formula, we pass outG as last parameter
    Ffact::template Eval< NEWINDS >(out, args..., outG);
  }

  template < class V, class GRADIN >
  using DiffT = Factorize< typename F::template DiffT< V, GRADIN >, G >;

};

template < class F, class G >
struct Factorize_Alias {
  using type = Factorize_Impl< F, G >;
};

// specialization in case G is of type Var : in this case there is no need for copying a Var into another Var,
// so we replace Factorize<F,Var> simply by F. This is usefull to avoid factorizing several times the same sub-formula
template < class F, int N, int DIM, int CAT >
struct Factorize_Alias< F, Var< N, DIM, CAT>> {
  using type = F;
};

// specialization in case G is of type IntConstant : again such a factorization is not interesting
template < class F, int N >
struct Factorize_Alias< F, IntConstant_Impl< N>> {
  using type = F;
};

// specialization in case G = F : not interesting either
template < class F >
struct Factorize_Alias< F, F > {
  using type = F;
};

// avoid specialization conflict..
template < int N, int DIM, int CAT >
struct Factorize_Alias< Var< N, DIM, CAT >, Var< N, DIM, CAT>> {
  using type = Var< N, DIM, CAT >;
};

// specializations in case G is a pack of types : we recursively factorize F by each subformula in the pack

// first specialization, when the pack is empty (termination case)
template < class F >
struct Factorize_Alias< F, univpack< >> {
  using type = F;
};

// then specialization when there is at least one element in the pack
// we use CondFactorize to factorize only when the type is present at least twice in the formula
template < class F, class G, class... GS >
struct Factorize_Alias< F, univpack< G, GS...>> {
  using type = Factorize< CondFactorize< F, G >, univpack< GS...>>;
};


// Auto factorization : factorize F by each of its subformulas
// N.B we comment out AutoFactorize and AllTypes in all code as of oct 2020 to speed up compile time
//template < class F >
//using AutoFactorize = Factorize< F, typename F::AllTypes >;

#define Factorize(F, G) KeopsNS<Factorize<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(G))>()
//#define AutoFactorize(F) KeopsNS<AutoFactorize<decltype(InvKeopsNS(F))>>()

}
