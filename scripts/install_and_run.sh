# setup_cgc_analysis.sh
#!/bin/bash

echo "Setting up CGC analysis environment..."

# Create directory structure
mkdir -p data/{planck,bao,sne,growth,lyalpha}
mkdir -p scripts results plots

# Install required packages
pip install numpy scipy matplotlib pandas emcee corner astropy requests

# Download example data files
echo "Downloading example data files..."

# Planck 2018 (simplified)
wget -O data/planck/planck_params.txt https://raw.githubusercontent.com/yourusername/cgc_data/main/planck_params.txt

# BAO measurements
wget -O data/bao/boss_dr12.txt https://raw.githubusercontent.com/yourusername/cgc_data/main/boss_dr12_bao.txt
wget -O data/bao/eboss_dr16.txt https://raw.githubusercontent.com/yourusername/cgc_data/main/eboss_dr16_bao.txt

# Growth measurements
wget -O data/growth/fsigma8.txt https://raw.githubusercontent.com/yourusername/cgc_data/main/fsigma8_data.txt

# H0 measurements
cat > data/H0_measurements.txt << EOF
# Measurement       H0      error
Planck2018         67.36   0.54
SH0ES2022          73.04   1.04
TRGB               69.8    1.9
Maser              73.9    3.0
EOF

echo "Setup complete!"
echo "To run analysis: python run_cgc_analysis.py"