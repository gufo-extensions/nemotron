{
  description = "Gufo with the NVIDIA Nemotron-H extension";
  inputs.gufo.url = "github:gufo-org/gufo/0260e1162f8713caee4d3f4ff77e017de7165f9c";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
