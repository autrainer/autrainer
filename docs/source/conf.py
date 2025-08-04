import datetime
import glob
import os
import sys
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
import toml


def build_authors(authors):
    return ", ".join([a.split(" <")[0] for a in authors])


pyroject = toml.load("../../pyproject.toml")

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath("../extensions"))

project = pyroject["tool"]["poetry"]["name"]
description = pyroject["tool"]["poetry"]["description"]
author = build_authors(pyroject["tool"]["poetry"]["authors"])
copyright = f"{datetime.datetime.now().year} (MIT) {author}"
release = pyroject["tool"]["poetry"]["version"]


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxarg.ext",
    "sphinxcontrib.jquery",
    "sphinx_hydra_configurations",
    "sphinx_lexers",
]

master_doc = "index"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = {"theme.css"}
html_js_files = ["external_links.js", "highlight_cli.js"]
autosummary_generate = True
autodoc_member_order = "bysource"
autoclass_content = "init"

html_theme_options = {
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_end": [],
    "collapse_navigation": False,
    "secondary_sidebar_items": ["page-toc"],
    "navbar_center": [],
    "logo": {
        "alt_text": f"{project} — {description}",
        "text": project,
        "image_dark": "_static/logo_dark.webp",
        "image_light": "_static/logo_light.webp",
    },
    "pygments_light_style": "catppuccin-latte",
    "pygments_dark_style": "catppuccin-mocha",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/autrainer/autrainer",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
html_sidebars = {"**": ["globaltoc"]}
html_favicon = "_static/favicon.ico"
html_title = f"{project} — {description}"


def set_custom_title(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict,
    doctree: Any,
):
    if pagename == app.config.master_doc:
        context["title"] = f"{project} — {description}"


class AddYamlDirective(Directive):
    required_arguments = 1  # subfolder path
    has_content = False

    def run(self):
        from sphinx.util import logging

        logger = logging.getLogger(__name__)

        env = self.state.document.settings.env
        doc_dir = os.path.dirname(env.doc2path(env.docname))
        target_dir = os.path.abspath(os.path.join(doc_dir, self.arguments[0]))
        logger.info(f"[add_yaml] Searching for YAML files in {target_dir} ...")

        if not os.path.isdir(target_dir):
            logger.warning(f"[add_yaml] Path does not exist: {target_dir}")
            return []

        # Collect YAML files recursively
        yaml_files = sorted(glob.glob(os.path.join(target_dir, "*.yaml")))

        output_dir = os.path.join(doc_dir, "_generated_yaml_pages")
        os.makedirs(output_dir, exist_ok=True)

        result = []
        for file_path in yaml_files:
            # need relative path for literalinclude
            # this is the parent dir of env.srcdir
            rel_path = os.path.join(
                os.path.pardir, os.path.relpath(file_path, start=env.srcdir)
            )

            literalinclude = nodes.literal_block("", "")
            literalinclude["classes"].append("code")
            literalinclude["source"] = file_path
            literalinclude["language"] = "yaml"
            literalinclude["linenos"] = False

            # Create directive string to re-parse
            caption = os.path.splitext(os.path.basename(file_path))[0]
            include_text = (
                f".. literalinclude:: {rel_path}\n"
                f"   :language: yaml\n"
                f"   :caption: {caption}\n"
                f"   :linenos:\n"
            )

            # Parse as RST
            include_lines = include_text.splitlines()
            self.state_machine.insert_input(include_lines, self.arguments[0])

        return result


def setup(app: Sphinx):
    app.connect("html-page-context", set_custom_title)
    app.add_directive("listyaml", AddYamlDirective)
