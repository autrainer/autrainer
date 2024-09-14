import os
from typing import Any, Dict, List, Optional

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.writers.html import HTML5Translator


class ConfigNode(nodes.General, nodes.Element): ...


def config_node_html(html: HTML5Translator, node: ConfigNode) -> None:
    cls = "literal-block-wrapper docutils container"
    html.body.append(html.starttag(node, "div", CLASS=cls))
    html.body.append(
        "<div class='code-block-caption'>"
        f"<span class='caption-text'>{node['caption']}</span>"
        f"<a class='headerlink' href='#{node['ids'][0]}' "
        "title='Link to this code'>#</a></div>"
    )


class ConfigurationsDirective(SphinxDirective):
    has_content = False
    option_spec = {
        "subdir": str,
        "configs": str,
        "headline": lambda _: True,
        "exact": lambda _: True,
    }

    def run(self) -> List[ConfigNode]:
        subdir = self.options.get("subdir", None)
        configs = self.options.get("configs", None)
        headline = self.options.get("headline", False)
        exact = self.options.get("exact", False)

        if configs is None:
            return []

        content = []
        for config in configs.split():
            content.extend(self._generate(subdir, config, headline, exact))

        return content

    def _generate(
        self,
        subdir: Optional[str],
        config: str,
        headline: bool = False,
        exact: bool = False,
    ) -> List[ConfigNode]:
        subdir = subdir or ""
        conf_dir = os.path.abspath(
            os.path.join(
                self.env.srcdir,
                f"../../autrainer-configurations/{subdir}",
            )
        )

        config_files = [
            f
            for f in os.listdir(conf_dir)
            if self._select_config(f, config, exact)
        ]
        config_files = sorted(config_files, key=lambda x: (x.lower(), len(x)))

        if not config_files:
            return []

        content = []
        if headline:
            content.append(nodes.paragraph("", "", nodes.strong(text=config)))

        for config_file in config_files:
            config_path = os.path.join(conf_dir, config_file)

            container_node = ConfigNode()
            container_node["caption"] = (
                f"conf/{subdir}/{config_file}"
                if subdir
                else f"conf/{config_file}"
            )
            config_id = f"default-{config_file.replace('.yaml', '').lower()}"
            container_node["ids"].append(config_id)

            with open(config_path, "r") as file:
                config_content = file.read()

            literal_node = nodes.literal_block(
                config_content,
                config_content,
                language="yaml",
                linenos=True,
            )

            container_node += literal_node
            content.append(container_node)

        return content

    @staticmethod
    def _select_config(f: str, config: str, exact: bool) -> bool:
        if exact:
            return f == f"{config}.yaml"
        return f.startswith(config) and f.endswith(".yaml")


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_node(
        ConfigNode,
        html=(
            config_node_html,
            lambda html, _: html.body.append("</div>"),
        ),
    )

    app.add_directive("configurations", ConfigurationsDirective)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
