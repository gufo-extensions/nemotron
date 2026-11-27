{
  description = "Gufo with the NVIDIA Nemotron-H extension";
  inputs.gufo.url = "github:gufo-org/gufo/684b716a567aee9e016cbe1e4fec22ae9344a7e8";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
