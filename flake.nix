{
  description = "Gufo with the NVIDIA Nemotron-H extension";
  inputs.gufo.url = "github:gufo-org/gufo/4a7d8c6d178fc6979e5ae38c07b40e1fe89dd1ce";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
