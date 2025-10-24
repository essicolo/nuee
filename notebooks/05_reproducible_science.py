import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 5: Reproducible Science and Open Research

    This chapter covers best practices for reproducible ecological research using 
    modern tools and workflows that ensure transparency, collaboration, and 
    scientific integrity.

    ## Learning Objectives
    - Understand principles of reproducible research
    - Master version control with Git for ecological projects
    - Create reproducible computational environments
    - Design robust data management workflows
    - Implement open science practices
    """
    )
    return


@app.cell
def __():
    # Essential imports for reproducible workflows
    import pandas as pd
    import numpy as np
    import hashlib
    import json
    import os
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
    print("✓ Reproducible science tools loaded")
    return datetime, hashlib, json, np, os, pd, warnings


@app.cell
def __():
    """
    ## Principles of Reproducible Research

    ### Core Components:

    1. **Transparency**: All methods, data, and code are openly available
    2. **Documentation**: Clear explanations of procedures and decisions
    3. **Version Control**: Track changes in code, data, and documents
    4. **Computational Environment**: Specify exact software versions
    5. **Data Management**: Organized, backed-up, and well-documented data
    6. **Automated Workflows**: Minimize manual steps and human error

    ### Benefits for Ecological Research:

    - **Scientific Integrity**: Others can verify and build upon your work
    - **Collaboration**: Easier to work with colleagues and share methods
    - **Efficiency**: Automated workflows save time on repetitive tasks
    - **Error Reduction**: Version control and documentation catch mistakes
    - **Career Advancement**: Reproducible work has higher impact and visibility
    - **Funding Requirements**: Many agencies now require data management plans
    """
    return


@app.cell
def __():
    """
    ## Project Organization Structure

    A well-organized project is the foundation of reproducible research.
    """
    
    # Demonstrate ideal project structure
    project_structure = """
    ecological_project/
    ├── README.md                 # Project overview and instructions
    ├── LICENSE                   # Usage rights and restrictions
    ├── requirements.txt          # Python package dependencies
    ├── environment.yml          # Conda environment specification
    ├── .gitignore              # Files to exclude from version control
    ├── data/
    │   ├── raw/                # Original, unmodified data
    │   ├── processed/          # Cleaned and transformed data
    │   └── metadata/           # Data descriptions and documentation
    ├── scripts/
    │   ├── 01_data_cleaning.py
    │   ├── 02_analysis.py
    │   └── 03_visualization.py
    ├── notebooks/
    │   ├── exploratory/        # Data exploration notebooks
    │   ├── analysis/          # Main analysis notebooks
    │   └── reports/           # Final reports and summaries
    ├── outputs/
    │   ├── figures/           # Generated plots and diagrams
    │   ├── tables/            # Summary tables and results
    │   └── models/            # Saved model objects
    ├── docs/
    │   ├── methods.md         # Detailed methodology
    │   ├── protocols.md       # Field and lab protocols
    │   └── references.bib     # Bibliography
    └── tests/
        ├── test_functions.py  # Unit tests for custom functions
        └── test_data.py       # Data validation tests
    """
    
    print("Recommended Project Structure:")
    print(project_structure)
    
    # Best practices for file naming
    naming_examples = {
        'Good': [
            '2024-01-15_site_survey_data.csv',
            'species_abundance_forest_sites.csv',
            'climate_data_processed_v2.csv',
            'figure_01_species_richness_elevation.png'
        ],
        'Bad': [
            'data.csv',
            'results final FINAL.xlsx', 
            'untitled1.py',
            'temp_file_delete_later.csv'
        ]
    }
    
    print("\nFile Naming Examples:")
    for category, examples in naming_examples.items():
        print(f"\n{category} Examples:")
        for example in examples:
            print(f"  - {example}")
    
    return naming_examples, project_structure


@app.cell
def __():
    """
    ## Version Control with Git

    Git is essential for tracking changes in your ecological research projects.
    """
    
    # Git workflow for ecological projects
    git_workflow = """
    Essential Git Commands for Ecological Research:
    
    # Initialize a new repository
    git init
    
    # Clone an existing repository
    git clone https://github.com/username/ecological-project.git
    
    # Check status of files
    git status
    
    # Add files to staging area
    git add data/processed/species_data.csv
    git add scripts/analysis.py
    git add .  # Add all changes
    
    # Commit changes with descriptive message
    git commit -m "Add species abundance analysis for forest sites"
    
    # View commit history
    git log --oneline
    
    # Create and switch to new branch
    git checkout -b feature/diversity-analysis
    
    # Merge branch back to main
    git checkout main
    git merge feature/diversity-analysis
    
    # Sync with remote repository
    git push origin main
    git pull origin main
    """
    
    print("Git Workflow for Ecological Research:")
    print(git_workflow)
    
    # Example .gitignore for ecological projects
    gitignore_content = """
    # .gitignore for Ecological Research Projects
    
    # Large data files (use Git LFS or external storage)
    *.csv
    *.xlsx
    *.nc
    *.tif
    *.shp
    
    # Processed outputs that can be regenerated
    outputs/figures/
    outputs/tables/
    
    # Temporary files
    .DS_Store
    Thumbs.db
    *.tmp
    *~
    
    # Python-specific
    __pycache__/
    *.pyc
    *.pyo
    *.egg-info/
    .pytest_cache/
    
    # Jupyter/Marimo checkpoints
    .ipynb_checkpoints/
    __marimo__/
    
    # Environment files (use requirements.txt instead)
    .env
    .venv/
    env/
    venv/
    
    # IDE files
    .vscode/
    .idea/
    *.swp
    
    # OS generated files
    .DS_Store
    .DS_Store?
    ._*
    .Spotlight-V100
    .Trashes
    ehthumbs.db
    Thumbs.db
    """
    
    print("\nExample .gitignore for ecological projects:")
    print(gitignore_content[:500] + "...")
    
    return git_workflow, gitignore_content


@app.cell
def __():
    """
    ## Data Management and Documentation
    """
    
    # Create example metadata for ecological dataset
    def create_metadata_template():
        metadata = {
            "dataset_info": {
                "title": "Forest Bird Community Survey Data",
                "description": "Point count surveys of bird communities across elevation gradients",
                "version": "1.0",
                "created_date": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "license": "CC-BY-4.0"
            },
            "spatial_coverage": {
                "geographic_extent": "Appalachian Mountains, Virginia, USA",
                "bounding_box": {
                    "north": 38.5,
                    "south": 37.0,
                    "east": -78.0,
                    "west": -80.0
                },
                "coordinate_system": "WGS84"
            },
            "temporal_coverage": {
                "start_date": "2023-05-01",
                "end_date": "2023-08-31",
                "frequency": "Monthly surveys during breeding season"
            },
            "methodology": {
                "sampling_method": "Point counts",
                "survey_duration": "10 minutes per point",
                "detection_radius": "50 meters",
                "time_of_day": "Dawn chorus (0.5 hours after sunrise)",
                "weather_conditions": "No rain, wind < 20 km/h"
            },
            "data_structure": {
                "site_id": {
                    "description": "Unique identifier for each survey site",
                    "type": "string",
                    "format": "SITE_XXX"
                },
                "species_code": {
                    "description": "4-letter species code following AOU standards",
                    "type": "string",
                    "example": "AMRO (American Robin)"
                },
                "abundance": {
                    "description": "Number of individuals detected",
                    "type": "integer",
                    "units": "count",
                    "minimum": 0
                },
                "elevation": {
                    "description": "Site elevation above sea level",
                    "type": "float",
                    "units": "meters",
                    "range": [300, 1500]
                }
            },
            "quality_control": {
                "data_validation": "Range checks, species code verification",
                "missing_data": "Coded as -999 for numeric, 'NA' for categorical",
                "outlier_detection": "Values >3 SD from mean flagged for review"
            },
            "contact_info": {
                "principal_investigator": "Dr. Jane Ecologist",
                "institution": "University of Ecology",
                "email": "j.ecologist@university.edu",
                "orcid": "0000-0000-0000-0000"
            }
        }
        return metadata
    
    metadata_example = create_metadata_template()
    
    print("Example Metadata Structure:")
    print(json.dumps(metadata_example, indent=2)[:1000] + "...")
    
    # Data validation example
    def validate_ecological_data(df, metadata):
        """
        Validate ecological data against metadata specifications
        """
        validation_results = {
            "total_records": len(df),
            "issues": []
        }
        
        # Check for required columns
        required_cols = list(metadata["data_structure"].keys())
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            validation_results["issues"].append(f"Missing columns: {missing_cols}")
        
        # Check data types and ranges
        for col, specs in metadata["data_structure"].items():
            if col in df.columns:
                # Check for missing values
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    validation_results["issues"].append(
                        f"{col}: {missing_count} missing values"
                    )
                
                # Check ranges for numeric columns
                if "range" in specs and col in df.columns:
                    min_val, max_val = specs["range"]
                    out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                    if len(out_of_range) > 0:
                        validation_results["issues"].append(
                            f"{col}: {len(out_of_range)} values out of range [{min_val}, {max_val}]"
                        )
        
        return validation_results
    
    # Example usage
    sample_data = pd.DataFrame({
        'site_id': ['SITE_001', 'SITE_002', 'SITE_003'],
        'species_code': ['AMRO', 'BCCH', 'WBNU'],
        'abundance': [3, 2, 1],
        'elevation': [450, 800, 1200]
    })
    
    validation = validate_ecological_data(sample_data, metadata_example)
    print(f"\nData validation results:")
    print(f"Total records: {validation['total_records']}")
    print(f"Issues found: {len(validation['issues'])}")
    
    return (
        create_metadata_template,
        metadata_example,
        sample_data,
        validate_ecological_data,
        validation,
    )


@app.cell
def __():
    """
    ## Computational Environment Management
    """
    
    # Example requirements.txt for ecological project
    requirements_example = """
    # Core data science packages
    pandas>=1.5.0
    numpy>=1.21.0
    scipy>=1.9.0
    matplotlib>=3.5.0
    
    # Ecological analysis
    nuee>=0.1.0
    scikit-learn>=1.1.0
    statsmodels>=0.13.0
    
    # Visualization
    holoviews>=1.15.0
    bokeh>=2.4.0
    
    # Geospatial (if needed)
    # geopandas>=0.11.0
    # rasterio>=1.3.0
    
    # Specific versions for reproducibility
    jupyter==1.0.0
    marimo>=0.10.0
    """
    
    # Example environment.yml for conda
    environment_yml = """
    name: ecological-analysis
    channels:
      - conda-forge
      - defaults
    dependencies:
      - python=3.10
      - pandas=1.5.3
      - numpy=1.24.0
      - scipy=1.10.0
      - matplotlib=3.6.0
      - jupyter=1.0.0
      - pip=23.0.0
      - pip:
        - nuee>=0.1.0
        - holoviews>=1.15.0
        - marimo>=0.10.0
    """
    
    print("Example requirements.txt:")
    print(requirements_example)
    print("\nExample environment.yml:")
    print(environment_yml)
    
    # Function to create environment snapshot
    def create_environment_snapshot():
        """
        Create a snapshot of the current computational environment
        """
        import sys
        import pkg_resources
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "packages": {}
        }
        
        # Get installed packages
        installed_packages = [d for d in pkg_resources.working_set]
        for package in installed_packages:
            snapshot["packages"][package.project_name] = package.version
        
        return snapshot
    
    # Create snapshot (would work in a full Python environment)
    try:
        env_snapshot = create_environment_snapshot()
        print(f"\nEnvironment snapshot created with {len(env_snapshot['packages'])} packages")
    except:
        print("\nEnvironment snapshot creation (limited in Pyodide environment)")
    
    return create_environment_snapshot, environment_yml, requirements_example


@app.cell
def __():
    """
    ## Automated Workflows and Scripts
    """
    
    # Example of a reproducible analysis script
    analysis_script_template = '''
    #!/usr/bin/env python3
    """
    Ecological Data Analysis Pipeline
    
    This script performs the complete analysis pipeline for forest bird community data.
    Run this script to reproduce all results in the manuscript.
    
    Usage:
        python scripts/analysis_pipeline.py
    
    Requirements:
        - All dependencies in requirements.txt
        - Raw data in data/raw/ directory
        - Metadata files in data/metadata/
    
    Outputs:
        - Processed data: data/processed/
        - Figures: outputs/figures/
        - Tables: outputs/tables/
        - Models: outputs/models/
    """
    
    import pandas as pd
    import numpy as np
    import logging
    from pathlib import Path
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    def main():
        """Main analysis pipeline"""
        
        logging.info("Starting ecological data analysis pipeline")
        
        # Step 1: Load and validate data
        logging.info("Step 1: Loading raw data")
        raw_data = load_raw_data()
        
        # Step 2: Clean and process data  
        logging.info("Step 2: Cleaning and processing data")
        processed_data = clean_data(raw_data)
        
        # Step 3: Perform statistical analyses
        logging.info("Step 3: Running statistical analyses")
        results = run_analyses(processed_data)
        
        # Step 4: Generate visualizations
        logging.info("Step 4: Creating visualizations")
        create_figures(processed_data, results)
        
        # Step 5: Save results
        logging.info("Step 5: Saving results")
        save_results(results)
        
        logging.info("Analysis pipeline completed successfully")
    
    if __name__ == "__main__":
        main()
    '''
    
    print("Example Analysis Script Template:")
    print(analysis_script_template[:800] + "...")
    
    # Example Makefile for automation
    makefile_example = """
    # Makefile for Ecological Research Project
    
    # Default target
    all: data analysis figures report
    
    # Data processing
    data: data/processed/species_data_clean.csv
    
    data/processed/species_data_clean.csv: data/raw/species_data.csv scripts/01_data_cleaning.py
    \tpython scripts/01_data_cleaning.py
    
    # Statistical analysis
    analysis: outputs/tables/results_summary.csv
    
    outputs/tables/results_summary.csv: data/processed/species_data_clean.csv scripts/02_analysis.py
    \tpython scripts/02_analysis.py
    
    # Generate figures
    figures: outputs/figures/species_richness_plot.png
    
    outputs/figures/species_richness_plot.png: outputs/tables/results_summary.csv scripts/03_visualization.py
    \tpython scripts/03_visualization.py
    
    # Generate final report
    report: outputs/final_report.html
    
    outputs/final_report.html: notebooks/final_analysis.py
    \tmarimo run notebooks/final_analysis.py --output outputs/final_report.html
    
    # Clean generated files
    clean:
    \trm -rf outputs/figures/*
    \trm -rf outputs/tables/*
    \trm -rf data/processed/*
    
    # Install dependencies
    install:
    \tpip install -r requirements.txt
    
    # Run tests
    test:
    \tpython -m pytest tests/
    
    .PHONY: all data analysis figures report clean install test
    """
    
    print("\nExample Makefile:")
    print(makefile_example[:600] + "...")
    
    return analysis_script_template, makefile_example


@app.cell
def __():
    """
    ## Data Provenance and Lineage Tracking
    """
    
    # Function to track data lineage
    def create_data_lineage_record(input_file, output_file, operation, parameters=None):
        """
        Create a record of data transformation for provenance tracking
        """
        lineage_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "input_file": input_file,
            "output_file": output_file,
            "parameters": parameters or {},
            "script_version": "v1.0",  # Could be git commit hash
            "user": os.getenv("USER", "unknown")
        }
        
        # Calculate file checksums for integrity verification
        def calculate_checksum(filepath):
            """Calculate MD5 checksum of file"""
            if os.path.exists(filepath):
                hash_md5 = hashlib.md5()
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()
            return None
        
        lineage_record["input_checksum"] = calculate_checksum(input_file)
        lineage_record["output_checksum"] = calculate_checksum(output_file)
        
        return lineage_record
    
    # Example usage
    lineage_example = create_data_lineage_record(
        input_file="data/raw/bird_survey_2023.csv",
        output_file="data/processed/bird_survey_clean.csv", 
        operation="data_cleaning",
        parameters={
            "remove_outliers": True,
            "outlier_threshold": 3.0,
            "missing_value_strategy": "interpolate"
        }
    )
    
    print("Data Lineage Record Example:")
    print(json.dumps(lineage_example, indent=2))
    
    # Data processing pipeline with lineage tracking
    def process_with_lineage(data, operation_name, parameters=None):
        """
        Process data while maintaining lineage information
        """
        # Record input state
        input_hash = hashlib.md5(str(data.values).encode()).hexdigest()
        
        # Perform operation (example: remove outliers)
        if operation_name == "remove_outliers":
            threshold = parameters.get("threshold", 3.0)
            z_scores = np.abs((data - data.mean()) / data.std())
            processed_data = data[z_scores < threshold]
        else:
            processed_data = data.copy()
        
        # Record output state
        output_hash = hashlib.md5(str(processed_data.values).encode()).hexdigest()
        
        # Create lineage record
        lineage = {
            "operation": operation_name,
            "timestamp": datetime.now().isoformat(),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "parameters": parameters,
            "input_shape": data.shape,
            "output_shape": processed_data.shape
        }
        
        return processed_data, lineage
    
    # Example data processing with lineage
    sample_data = pd.Series([1, 2, 3, 100, 5, 6, 7, 8, 9, 10])  # Contains outlier
    processed, lineage_info = process_with_lineage(
        sample_data, 
        "remove_outliers", 
        {"threshold": 2.0}
    )
    
    print(f"\nData processing example:")
    print(f"Original shape: {lineage_info['input_shape']}")
    print(f"Processed shape: {lineage_info['output_shape']}")
    print(f"Operation: {lineage_info['operation']}")
    
    return (
        calculate_checksum,
        create_data_lineage_record,
        lineage_example,
        lineage_info,
        process_with_lineage,
        processed,
    )


@app.cell
def __():
    """
    ## Open Science Practices
    """
    
    # Guidelines for open ecological research
    open_science_checklist = {
        "Data Sharing": [
            "✓ Deposit data in appropriate repositories (Dryad, Figshare, Zenodo)",
            "✓ Use standardized formats (CSV, NetCDF, GeoTIFF)",
            "✓ Include comprehensive metadata",
            "✓ Assign DOIs to datasets",
            "✓ Follow FAIR principles (Findable, Accessible, Interoperable, Reusable)"
        ],
        "Code Sharing": [
            "✓ Use public version control (GitHub, GitLab)",
            "✓ Include clear README with installation instructions",
            "✓ Add appropriate license (MIT, GPL, Apache)",
            "✓ Create releases/tags for publication versions",
            "✓ Document code with comments and docstrings"
        ],
        "Publication": [
            "✓ Preprint on bioRxiv or EcoEvoRxiv",
            "✓ Link to data and code repositories",
            "✓ Use persistent identifiers (ORCID, DOI)",
            "✓ Submit to open access journals when possible",
            "✓ Share on social media and professional networks"
        ],
        "Collaboration": [
            "✓ Use collaborative platforms (GitHub, OSF)",
            "✓ Document contribution guidelines",
            "✓ Establish clear authorship criteria",
            "✓ Regular team meetings and progress updates",
            "✓ Shared project management tools"
        ]
    }
    
    print("Open Science Checklist for Ecological Research:")
    for category, items in open_science_checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")
    
    # Example README template for ecological projects
    readme_template = '''
    # Forest Bird Community Analysis
    
    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)
    [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
    
    ## Overview
    
    This repository contains code and data for analyzing forest bird community 
    responses to elevation gradients in the Appalachian Mountains.
    
    ## Citation
    
    If you use this code or data, please cite:
    
    > Smith, J. et al. (2024). Elevation effects on forest bird communities. 
    > Journal of Ecology, 112(3), 456-478. https://doi.org/10.1111/1365-2745.14234
    
    ## Data
    
    Raw data is available at: https://doi.org/10.5061/dryad.abc123
    
    - `bird_surveys.csv`: Point count survey data
    - `site_characteristics.csv`: Environmental variables for each site
    - `species_traits.csv`: Functional trait data for bird species
    
    ## Installation
    
    ```bash
    # Clone repository
    git clone https://github.com/username/forest-bird-analysis.git
    cd forest-bird-analysis
    
    # Install dependencies
    pip install -r requirements.txt
    ```
    
    ## Usage
    
    ```bash
    # Run complete analysis pipeline
    python scripts/analysis_pipeline.py
    
    # Or run individual steps
    python scripts/01_data_cleaning.py
    python scripts/02_analysis.py
    python scripts/03_visualization.py
    ```
    
    ## Repository Structure
    
    ```
    forest-bird-analysis/
    ├── data/
    │   ├── raw/              # Original survey data
    │   └── processed/        # Cleaned data ready for analysis
    ├── scripts/              # Analysis scripts
    ├── notebooks/            # Marimo notebooks for exploration
    ├── outputs/              # Generated figures and tables
    └── docs/                 # Additional documentation
    ```
    
    ## Contributing
    
    We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
    
    ## License
    
    This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
    
    ## Contact
    
    - Dr. Jane Smith (jane.smith@university.edu)
    - ORCID: 0000-0000-0000-0000
    '''
    
    print(f"\nExample README.md template:")
    print(readme_template[:800] + "...")
    
    return open_science_checklist, readme_template


@app.cell
def __():
    """
    ## Testing and Quality Assurance
    """
    
    # Example unit tests for ecological functions
    test_example = '''
    # tests/test_diversity_functions.py
    
    import pytest
    import numpy as np
    import pandas as pd
    from scripts.diversity_functions import shannon_diversity, simpson_diversity
    
    class TestDiversityFunctions:
        """Test suite for biodiversity calculation functions"""
        
        def test_shannon_diversity_known_values(self):
            """Test Shannon diversity with known expected values"""
            # Equal abundances should give maximum diversity
            equal_abundances = [10, 10, 10, 10]
            expected = np.log(4)  # ln(4) for 4 equally abundant species
            result = shannon_diversity(equal_abundances)
            assert abs(result - expected) < 0.001
        
        def test_shannon_diversity_single_species(self):
            """Test Shannon diversity with single species (should be 0)"""
            single_species = [100, 0, 0, 0]
            result = shannon_diversity(single_species)
            assert result == 0
        
        def test_shannon_diversity_empty_input(self):
            """Test Shannon diversity with empty input"""
            with pytest.raises(ValueError):
                shannon_diversity([])
        
        def test_simpson_diversity_range(self):
            """Test that Simpson diversity is always between 0 and 1"""
            test_cases = [
                [10, 5, 3, 2, 1],
                [100, 0, 0, 0],
                [1, 1, 1, 1, 1, 1]
            ]
            
            for abundances in test_cases:
                result = simpson_diversity(abundances)
                assert 0 <= result <= 1
        
        def test_diversity_with_dataframe(self):
            """Test diversity functions with pandas DataFrame input"""
            df = pd.DataFrame({
                'species_a': [10, 5, 8],
                'species_b': [5, 10, 3], 
                'species_c': [2, 3, 5]
            })
            
            # Should work with DataFrame rows
            for _, row in df.iterrows():
                shannon_result = shannon_diversity(row.values)
                simpson_result = simpson_diversity(row.values)
                
                assert shannon_result >= 0
                assert 0 <= simpson_result <= 1
    
    # Run tests with: python -m pytest tests/test_diversity_functions.py -v
    '''
    
    print("Example Unit Tests for Ecological Functions:")
    print(test_example[:1000] + "...")
    
    # Data validation tests
    data_validation_test = '''
    # tests/test_data_validation.py
    
    import pytest
    import pandas as pd
    import numpy as np
    
    class TestDataValidation:
        """Test suite for ecological data validation"""
        
        def test_species_codes_format(self):
            """Test that species codes follow 4-letter format"""
            valid_codes = ['AMRO', 'BCCH', 'WBNU']
            invalid_codes = ['AMR', 'AMERICANROBIN', 'amro']
            
            for code in valid_codes:
                assert len(code) == 4
                assert code.isupper()
                assert code.isalpha()
            
            for code in invalid_codes:
                assert not (len(code) == 4 and code.isupper() and code.isalpha())
        
        def test_abundance_values(self):
            """Test that abundance values are non-negative integers"""
            valid_abundances = [0, 1, 5, 100]
            invalid_abundances = [-1, 3.5, np.inf, np.nan]
            
            for abundance in valid_abundances:
                assert abundance >= 0
                assert isinstance(abundance, (int, np.integer))
            
            for abundance in invalid_abundances:
                if not np.isnan(abundance):
                    assert not (abundance >= 0 and isinstance(abundance, (int, np.integer)))
        
        def test_coordinate_ranges(self):
            """Test that coordinates are within valid ranges"""
            # Example for Virginia, USA
            valid_coords = [
                {'lat': 37.5, 'lon': -79.0},
                {'lat': 38.2, 'lon': -78.5}
            ]
            
            invalid_coords = [
                {'lat': 91.0, 'lon': -79.0},   # Latitude out of range
                {'lat': 37.5, 'lon': 181.0}   # Longitude out of range
            ]
            
            for coord in valid_coords:
                assert -90 <= coord['lat'] <= 90
                assert -180 <= coord['lon'] <= 180
    '''
    
    print("\nExample Data Validation Tests:")
    print(data_validation_test[:800] + "...")
    
    return data_validation_test, test_example


@app.cell
def __():
    """
    ## Continuous Integration and Automation
    """
    
    # Example GitHub Actions workflow
    github_actions_workflow = '''
    # .github/workflows/analysis.yml
    
    name: Ecological Data Analysis
    
    on:
      push:
        branches: [ main, develop ]
      pull_request:
        branches: [ main ]
      schedule:
        # Run weekly to check for issues
        - cron: '0 0 * * 0'
    
    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: [3.9, 3.10, 3.11]
    
        steps:
        - uses: actions/checkout@v3
        
        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v3
          with:
            python-version: ${{ matrix.python-version }}
        
        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            pip install pytest pytest-cov
        
        - name: Run tests
          run: |
            pytest tests/ --cov=scripts/ --cov-report=xml
        
        - name: Upload coverage to Codecov
          uses: codecov/codecov-action@v3
          with:
            file: ./coverage.xml
        
        - name: Run data validation
          run: |
            python scripts/validate_data.py
        
        - name: Generate analysis report
          run: |
            python scripts/analysis_pipeline.py
        
        - name: Archive results
          uses: actions/upload-artifact@v3
          with:
            name: analysis-results-${{ matrix.python-version }}
            path: outputs/
    '''
    
    print("Example GitHub Actions Workflow:")
    print(github_actions_workflow[:800] + "...")
    
    # Pre-commit hooks configuration
    precommit_config = '''
    # .pre-commit-config.yaml
    
    repos:
    -   repo: https://github.com/pre-commit/pre-commit-hooks
        rev: v4.4.0
        hooks:
        -   id: trailing-whitespace
        -   id: end-of-file-fixer
        -   id: check-yaml
        -   id: check-added-large-files
            args: ['--maxkb=1000']  # Prevent large data files
        -   id: check-json
        -   id: check-merge-conflict
    
    -   repo: https://github.com/psf/black
        rev: 23.3.0
        hooks:
        -   id: black
            language_version: python3
    
    -   repo: https://github.com/pycqa/flake8
        rev: 6.0.0
        hooks:
        -   id: flake8
            args: [--max-line-length=88]
    
    -   repo: local
        hooks:
        -   id: data-validation
            name: Validate ecological data
            entry: python scripts/validate_data.py
            language: system
            pass_filenames: false
            files: ^data/
    '''
    
    print("\nExample Pre-commit Configuration:")
    print(precommit_config[:600] + "...")
    
    return github_actions_workflow, precommit_config


@app.cell
def __():
    """
    ## Reproducibility Checklist

    Before publishing your ecological research, ensure you've addressed these components:

    ### 📊 Data Management
    - [ ] Data stored in open repository with DOI
    - [ ] Comprehensive metadata provided
    - [ ] Raw data preserved and documented
    - [ ] Data processing steps clearly documented
    - [ ] File formats are non-proprietary and widely supported

    ### 💻 Code and Analysis
    - [ ] All analysis code available in public repository
    - [ ] Code is well-commented and documented
    - [ ] Dependencies clearly specified (requirements.txt)
    - [ ] Analysis can be run with single command
    - [ ] Random seeds set for reproducible results

    ### 📚 Documentation
    - [ ] README file with clear instructions
    - [ ] Methods section describes computational approach
    - [ ] Software versions reported
    - [ ] Hardware/platform specifications noted
    - [ ] Contact information provided

    ### 🔄 Version Control
    - [ ] Project uses Git version control
    - [ ] Meaningful commit messages
    - [ ] Tagged release for publication
    - [ ] No large data files in Git repository
    - [ ] Appropriate .gitignore file

    ### 🧪 Quality Assurance
    - [ ] Unit tests for custom functions
    - [ ] Data validation tests
    - [ ] Continuous integration set up
    - [ ] Code follows style guidelines
    - [ ] Peer review of code and methods

    ### 🌐 Open Science
    - [ ] Preprint posted before submission
    - [ ] Open access publication when possible
    - [ ] ORCID IDs for all authors
    - [ ] Appropriate Creative Commons license
    - [ ] Results shared on social media/conferences
    """
    
    checklist_summary = {
        "Essential": [
            "Public code repository with clear documentation",
            "Open data with comprehensive metadata", 
            "Reproducible computational environment",
            "Version control throughout project lifecycle"
        ],
        "Recommended": [
            "Automated testing and validation",
            "Continuous integration workflows",
            "Pre-commit hooks for quality control",
            "Data and code DOIs for citation"
        ],
        "Advanced": [
            "Containerized environments (Docker)",
            "Workflow management systems",
            "Automated report generation",
            "Integration with lab information systems"
        ]
    }
    
    print("Reproducibility Implementation Levels:")
    for level, items in checklist_summary.items():
        print(f"\n{level}:")
        for item in items:
            print(f"  • {item}")
    
    return checklist_summary,


@app.cell
def __():
    """
    ## Summary

    In this chapter, we covered essential practices for reproducible ecological research:

    ✓ **Project organization**: Structured directories and file naming
    ✓ **Version control**: Git workflows for ecological projects  
    ✓ **Data management**: Metadata, validation, and provenance tracking
    ✓ **Environment management**: Reproducible computational setups
    ✓ **Automation**: Scripts, pipelines, and continuous integration
    ✓ **Open science**: Data sharing, code repositories, and collaboration
    ✓ **Quality assurance**: Testing, validation, and peer review
    ✓ **Documentation**: README files, methods, and communication

    **Next chapter**: Biostatistics for ecological data analysis

    **Key principles**:
    - **Transparency**: Make everything openly available
    - **Documentation**: Explain every step and decision
    - **Automation**: Reduce manual errors and increase efficiency
    - **Validation**: Test data and code rigorously
    - **Collaboration**: Use tools that facilitate teamwork
    - **Persistence**: Use permanent identifiers and stable repositories

    **Tools covered**:
    - **Git**: Version control and collaboration
    - **GitHub**: Repository hosting and project management  
    - **Requirements.txt/environment.yml**: Dependency management
    - **GitHub Actions**: Continuous integration
    - **Pre-commit**: Automated quality checks
    - **Pytest**: Testing framework
    """
    print("✓ Chapter 5 complete! Ready for biostatistical analysis.")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()