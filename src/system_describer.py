#!/usr/bin/env python3
"""
Dynamic System Self-Describer
Generates up-to-date documentation of your codebase for LLM context.
"""

import ast
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class FunctionInfo:
    name: str
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    line_number: int
    is_async: bool = False


@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    methods: List[FunctionInfo]
    docstring: Optional[str]
    line_number: int


@dataclass
class ModuleInfo:
    filepath: str
    imports: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    line_count: int
    docstring: Optional[str]


class CodeAnalyzer(ast.NodeVisitor):
    """Extracts structured information from Python AST."""

    def __init__(self):
        self.imports = []
        self.functions = []
        self.classes = []
        self.module_docstring = None

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Only capture top-level functions (not methods)
        if isinstance(node, ast.FunctionDef) and not isinstance(
            getattr(self, "_current_class", None), ast.ClassDef
        ):
            func_info = self._extract_function_info(node)
            self.functions.append(func_info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if not isinstance(getattr(self, "_current_class", None), ast.ClassDef):
            func_info = self._extract_function_info(node, is_async=True)
            self.functions.append(func_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._current_class = node

        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(
                    self._extract_function_info(
                        item, is_async=isinstance(item, ast.AsyncFunctionDef)
                    )
                )

        class_info = ClassInfo(
            name=node.name,
            bases=[self._get_name(base) for base in node.bases],
            methods=methods,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
        )
        self.classes.append(class_info)

        self._current_class = None
        self.generic_visit(node)

    def visit_Module(self, node):
        self.module_docstring = ast.get_docstring(node)
        self.generic_visit(node)

    def _extract_function_info(self, node, is_async=False):
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            args.append(arg_str)

        returns = None
        if node.returns:
            returns = self._get_name(node.returns)

        return FunctionInfo(
            name=node.name,
            args=args,
            returns=returns,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
            is_async=is_async,
        )

    def _get_name(self, node):
        """Extract name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return str(node)


class SystemDescriber:
    """Main system documentation generator."""

    def __init__(self, root_dir: str, focus_dirs: List[str] = None):
        self.root_dir = Path(root_dir)
        self.focus_dirs = focus_dirs or ["core", "diagnostics"]
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)

    def analyze(self):
        """Scan and analyze all Python files in focus directories."""
        for focus_dir in self.focus_dirs:
            dir_path = self.root_dir / focus_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                self._analyze_file(py_file)

        # Also analyze key files in root
        root_files = [
            "integrated_system_production.py",
            "train_probabilistic_models.py",
            "config.py",
            "logging_config.py",
        ]
        for filename in root_files:
            filepath = self.root_dir / filename
            if filepath.exists():
                self._analyze_file(filepath)

    def _analyze_file(self, filepath: Path):
        """Analyze a single Python file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
                tree = ast.parse(source, filename=str(filepath))

            analyzer = CodeAnalyzer()
            analyzer.visit(tree)

            # Calculate relative path
            try:
                rel_path = filepath.relative_to(self.root_dir)
            except ValueError:
                rel_path = filepath

            module_info = ModuleInfo(
                filepath=str(rel_path),
                imports=list(set(analyzer.imports)),  # deduplicate
                functions=analyzer.functions,
                classes=analyzer.classes,
                line_count=len(source.splitlines()),
                docstring=analyzer.module_docstring,
            )

            self.modules[str(rel_path)] = module_info

            # Build dependency graph
            for imp in analyzer.imports:
                self.dependency_graph[str(rel_path)].add(imp)

        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")

    def generate_markdown(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive markdown documentation."""
        lines = [
            f"# System Architecture Documentation",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Overview",
            f"- Total modules analyzed: {len(self.modules)}",
            f"- Total lines of code: {sum(m.line_count for m in self.modules.values())}",
            "",
            "## Module Structure",
            "",
        ]

        # Group by directory
        by_dir = defaultdict(list)
        for path, module in self.modules.items():
            dir_name = str(Path(path).parent) if "/" in path else "root"
            by_dir[dir_name].append((path, module))

        for dir_name in sorted(by_dir.keys()):
            lines.append(f"### {dir_name}/")
            lines.append("")

            for path, module in sorted(by_dir[dir_name]):
                lines.append(f"#### `{Path(path).name}` ({module.line_count} lines)")

                if module.docstring:
                    lines.append(f"*{module.docstring.split(chr(10))[0]}*")
                    lines.append("")

                # Classes
                if module.classes:
                    lines.append("**Classes:**")
                    for cls in module.classes:
                        base_str = (
                            f" extends {', '.join(cls.bases)}" if cls.bases else ""
                        )
                        lines.append(f"- `{cls.name}{base_str}`")
                        if cls.docstring:
                            lines.append(f"  - {cls.docstring.split(chr(10))[0]}")

                        # Key methods (skip private/dunder except __init__)
                        public_methods = [
                            m
                            for m in cls.methods
                            if not m.name.startswith("_") or m.name == "__init__"
                        ]
                        if public_methods:
                            for method in public_methods[:5]:  # Limit to first 5
                                args_str = ", ".join(
                                    method.args[:3]
                                )  # Limit args shown
                                if len(method.args) > 3:
                                    args_str += ", ..."
                                lines.append(f"    - `{method.name}({args_str})`")
                    lines.append("")

                # Functions
                if module.functions:
                    lines.append("**Functions:**")
                    for func in module.functions:
                        if not func.name.startswith("_"):  # Skip private
                            args_str = ", ".join(func.args[:3])
                            if len(func.args) > 3:
                                args_str += ", ..."
                            ret_str = f" -> {func.returns}" if func.returns else ""
                            lines.append(f"- `{func.name}({args_str}){ret_str}`")
                    lines.append("")

                # Key imports (exclude stdlib)
                non_stdlib = [
                    imp
                    for imp in module.imports
                    if "." in imp or imp in ["numpy", "pandas", "xgboost", "sklearn"]
                ]
                if non_stdlib:
                    lines.append(
                        f"**Key Dependencies:** {', '.join(sorted(set(non_stdlib))[:8])}"
                    )
                    lines.append("")

                lines.append("")

        # Dependency relationships
        lines.extend(["## Module Dependencies", "", "Key internal dependencies:", ""])

        internal_deps = {}
        for module, imports in self.dependency_graph.items():
            internal = [
                imp
                for imp in imports
                if any(imp.startswith(dir_) for dir_ in self.focus_dirs)
            ]
            if internal:
                internal_deps[Path(module).name] = internal

        for module, deps in sorted(internal_deps.items()):
            lines.append(f"- `{module}` → {', '.join(deps)}")

        markdown = "\n".join(lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown)
            print(f"Documentation written to {output_file}")

        return markdown

    def generate_json(self, output_file: Optional[str] = None) -> str:
        """Generate JSON representation for programmatic access."""
        data = {
            "generated": datetime.now().isoformat(),
            "summary": {
                "total_modules": len(self.modules),
                "total_lines": sum(m.line_count for m in self.modules.values()),
                "total_classes": sum(len(m.classes) for m in self.modules.values()),
                "total_functions": sum(len(m.functions) for m in self.modules.values()),
            },
            "modules": {
                path: {
                    "line_count": module.line_count,
                    "docstring": module.docstring,
                    "imports": module.imports,
                    "classes": [
                        {
                            "name": cls.name,
                            "bases": cls.bases,
                            "docstring": cls.docstring,
                            "methods": [m.name for m in cls.methods],
                        }
                        for cls in module.classes
                    ],
                    "functions": [f.name for f in module.functions],
                }
                for path, module in self.modules.items()
            },
            "dependencies": {k: list(v) for k, v in self.dependency_graph.items()},
        }

        json_str = json.dumps(data, indent=2)

        if output_file:
            with open(output_file, "w") as f:
                f.write(json_str)
            print(f"JSON documentation written to {output_file}")

        return json_str

    def generate_llm_context(self, output_file: Optional[str] = None) -> str:
        """Generate concise, LLM-optimized description."""
        lines = [
            "# System Context for LLM",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Quick Stats",
            f"- Modules: {len(self.modules)} | Lines: {sum(m.line_count for m in self.modules.values())}",
            f"- Classes: {sum(len(m.classes) for m in self.modules.values())} | Functions: {sum(len(m.functions) for m in self.modules.values())}",
            "",
            "## Core Architecture",
            "",
        ]

        # Focus on the most important modules
        important_modules = [
            "integrated_system_production.py",
            "train_probabilistic_models.py",
            "config.py",
        ]

        core_dir = [m for m in self.modules.items() if "core/" in m[0]]

        lines.append("### Entry Points")
        for path, module in self.modules.items():
            if any(imp in path for imp in important_modules):
                lines.append(f"- **{Path(path).name}**: {module.line_count}L")
                if module.docstring:
                    lines.append(f"  {module.docstring.split(chr(10))[0]}")
        lines.append("")

        lines.append("### Core Modules (core/)")
        for path, module in sorted(core_dir):
            filename = Path(path).name
            classes_str = (
                ", ".join(c.name for c in module.classes)
                if module.classes
                else "utilities"
            )
            lines.append(f"- **{filename}**: {classes_str}")
        lines.append("")

        # Highlight key classes
        lines.append("## Key Classes & Their Roles")
        for path, module in self.modules.items():
            for cls in module.classes:
                if len(cls.methods) >= 3:  # Substantial classes only
                    lines.append(f"- **{cls.name}** ({Path(path).name})")
                    if cls.docstring:
                        lines.append(f"  Purpose: {cls.docstring.split(chr(10))[0]}")
                    key_methods = [
                        m.name for m in cls.methods if not m.name.startswith("_")
                    ][:4]
                    lines.append(f"  Methods: {', '.join(key_methods)}")

        context = "\n".join(lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(context)
            print(f"LLM context written to {output_file}")

        return context


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate system documentation")
    parser.add_argument("--root", default=".", help="Root directory of project")
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "llm", "all"],
        default="llm",
        help="Output format",
    )
    parser.add_argument("--output", help="Output file (default: print to stdout)")
    parser.add_argument(
        "--focus",
        nargs="+",
        default=["core", "diagnostics"],
        help="Directories to focus analysis on",
    )

    args = parser.parse_args()

    describer = SystemDescriber(args.root, args.focus)
    describer.analyze()

    if args.format == "markdown" or args.format == "all":
        output = args.output or "SYSTEM_DESCRIPTION.md"
        describer.generate_markdown(
            output if args.format != "all" else "SYSTEM_DESCRIPTION.md"
        )

    if args.format == "json" or args.format == "all":
        output = args.output or "system_description.json"
        describer.generate_json(
            output if args.format != "all" else "system_description.json"
        )

    if args.format == "llm" or args.format == "all":
        output = args.output or "LLM_CONTEXT.md"
        result = describer.generate_llm_context(
            output if args.format != "all" else "LLM_CONTEXT.md"
        )
        if args.format == "llm" and not args.output:
            print(result)


if __name__ == "__main__":
    main()
