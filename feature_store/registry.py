"""
Feature Registry
================
Registry for feature definitions and metadata.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class FeatureMetadata:
    """Metadata for a single feature"""
    name: str
    dtype: str
    description: str = ""
    category: str = ""  # e.g., "microstructure", "momentum", "regime"
    source: str = ""  # "computed", "external", "derived"
    dependencies: list[str] = field(default_factory=list)  # Features this depends on
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    deprecated: bool = False
    deprecation_reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureMetadata":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class FeatureRegistry:
    """Registry for feature definitions"""
    
    def __init__(self, registry_path: str | Path | None = None):
        self.registry_path = Path(registry_path) if registry_path else None
        self.features: dict[str, FeatureMetadata] = {}
        self._lock = threading.Lock()
        
        if self.registry_path and self.registry_path.exists():
            self.load()
    
    def register(
        self,
        name: str,
        dtype: pl.DataType | str,
        description: str = "",
        category: str = "",
        source: str = "computed",
        dependencies: list[str] | None = None,
        tags: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> FeatureMetadata:
        """Register a feature"""
        with self._lock:
            if name in self.features and not overwrite:
                raise ValueError(f"Feature {name} already registered")
            
            dtype_str = str(dtype) if isinstance(dtype, pl.DataType) else dtype
            
            metadata = FeatureMetadata(
                name=name,
                dtype=dtype_str,
                description=description,
                category=category,
                source=source,
                dependencies=dependencies or [],
                tags=tags or {},
            )
            
            self.features[name] = metadata
            
            if self.registry_path:
                self.save()
            
            return metadata
    
    def register_from_dataframe(
        self,
        df: pl.DataFrame,
        category: str = "",
        source: str = "computed",
        overwrite: bool = False,
    ) -> dict[str, FeatureMetadata]:
        """Register all columns from a DataFrame"""
        registered = {}
        for col in df.columns:
            meta = self.register(
                name=col,
                dtype=df.schema[col],
                category=category,
                source=source,
                overwrite=overwrite,
            )
            registered[col] = meta
        return registered
    
    def get(self, name: str) -> FeatureMetadata | None:
        """Get feature metadata"""
        return self.features.get(name)
    
    def list_features(
        self,
        category: str | None = None,
        source: str | None = None,
        include_deprecated: bool = False,
    ) -> list[FeatureMetadata]:
        """List features with optional filters"""
        with self._lock:
            features = list(self.features.values())
            
            if category:
                features = [f for f in features if f.category == category]
            if source:
                features = [f for f in features if f.source == source]
            if not include_deprecated:
                features = [f for f in features if not f.deprecated]
            
            return sorted(features, key=lambda f: f.name)
    
    def get_categories(self) -> list[str]:
        """Get all categories"""
        return sorted(set(f.category for f in self.features.values() if f.category))
    
    def get_dependencies(self, name: str) -> list[str]:
        """Get dependencies for a feature"""
        meta = self.features.get(name)
        return meta.dependencies if meta else []
    
    def get_dependents(self, name: str) -> list[str]:
        """Get features that depend on this feature"""
        dependents = []
        for feat in self.features.values():
            if name in feat.dependencies:
                dependents.append(feat.name)
        return dependents
    
    def deprecate(self, name: str, reason: str):
        """Deprecate a feature"""
        with self._lock:
            if name in self.features:
                self.features[name].deprecated = True
                self.features[name].deprecation_reason = reason
                self.features[name].updated_at = datetime.now()
                if self.registry_path:
                    self.save()
    
    def save(self):
        """Save registry to disk"""
        if not self.registry_path:
            return
        
        with self._lock:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "features": {k: v.to_dict() for k, v in self.features.items()},
            }
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
    
    def load(self):
        """Load registry from disk"""
        if not self.registry_path or not self.registry_path.exists():
            return
        
        with self._lock:
            with open(self.registry_path) as f:
                data = json.load(f)
            
            self.features = {
                k: FeatureMetadata.from_dict(v) for k, v in data.get("features", {}).items()
            }
    
    def export_catalog(self, output_path: str | Path):
        """Export feature catalog as markdown"""
        output_path = Path(output_path)
        
        lines = [
            "# Feature Catalog",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Features: {len(self.features)}",
            "",
        ]
        
        for category in self.get_categories():
            features = self.list_features(category=category)
            if not features:
                continue
            
            lines.append(f"## {category}")
            lines.append("")
            lines.append("| Feature | Type | Description | Source | Dependencies |")
            lines.append("|---------|------|-------------|--------|--------------|")
            
            for feat in features:
                deps = ", ".join(feat.dependencies) if feat.dependencies else "-"
                lines.append(f"| {feat.name} | {feat.dtype} | {feat.description} | {feat.source} | {deps} |")
            
            lines.append("")
        
        output_path.write_text("\n".join(lines))


# Global registry instance
_global_registry: FeatureRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> FeatureRegistry:
    """Get global registry instance"""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = FeatureRegistry()
        return _global_registry


def set_registry(registry: FeatureRegistry):
    """Set global registry instance"""
    global _global_registry
    with _registry_lock:
        _global_registry = registry