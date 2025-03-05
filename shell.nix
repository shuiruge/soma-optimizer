# shell.nix
with import <nixpkgs> { };
let
  pythonPackages = python311Packages;
in pkgs.mkShell rec {
  name = "soma-optimizer";
  buildInputs = [
    pythonPackages.python
    pythonPackages.ipykernel
    pythonPackages.optax
  ];
}
