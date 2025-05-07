# LanceDB Documentation

This repository contains the official documentation for LanceDB, a modern vector database designed for AI applications. The documentation covers:

- Getting started guides
- Core concepts and architecture
- API reference
- Best practices and tutorials
- Integration guides
- Performance optimization
- Troubleshooting

The documentation is built using MkDocs Material to provide an optimal reading experience for developers.

## How to build the site

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lancedb/docs.git
cd docs
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install MkDocs and Material theme:
```bash
pip install mkdocs-material
```

### Development

To start the development server:

```bash
mkdocs serve
```

The site will be available at `http://127.0.0.1:8000`

### Building for Production

To build the site for production:

```bash
mkdocs build
```

The built files will be available in the `site` directory.

### Deployment

The documentation site is automatically deployed when changes are pushed to the main branch. The deployment process is handled by our CI/CD pipeline.

### Additional Configuration

The site configuration is managed through `mkdocs.yml`. Key features include:

- Navigation structure
- Theme customization
- Search functionality
- Versioning
- Analytics integration

For more information about customizing the site, refer to the [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/).
