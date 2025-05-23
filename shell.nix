# shell.nix
with import <nixpkgs> { };
let
  pythonPackages = python312Packages;
in pkgs.mkShell rec {
  name = "soma-optimizer";
  buildInputs = [
    pythonPackages.python
    #pythonPackages.ipykernel
    pythonPackages.optax

    # for testing
    pythonPackages.numpy
    pythonPackages.jax
    pythonPackages.flax
    pythonPackages.matplotlib
    # for loading datasets.
    pythonPackages.scikit-learn
    pythonPackages.torch
    pythonPackages.torchvision
  ];
}
