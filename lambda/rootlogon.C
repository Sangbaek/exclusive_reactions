
{
  gInterpreter->AddIncludePath("/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/hipo/4.5.0/root_6.38.04/include");

  gInterpreter->ProcessLine("#include \"hipo4/RHipoDS.hxx\"");
  gInterpreter->ProcessLine("#include \"hipo4/RIguanaDS.hxx\"");
}
