# data/database.py

import sqlite3
import json
import os
from typing import List, Optional
from .models import Product

class ProductDatabase:
    """SQLite database for product configurations"""
    
    def __init__(self, db_path: str = "data/products.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_date TEXT,
                    modified_date TEXT
                )
            ''')
    
    def save_product(self, product: Product):
        """Save or update product configuration"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO products (product_id, name, config_json, created_date, modified_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                product.product_id,
                product.name,
                json.dumps(product.to_dict()),
                product.created_date,
                product.modified_date
            ))
    
    def load_product(self, product_id: str) -> Optional[Product]:
        """Load product by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT config_json FROM products WHERE product_id = ?',
                (product_id,)
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return Product.from_dict(data)
        return None
    
    def list_products(self) -> List[dict]:
        """List all available products"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT product_id, name, modified_date FROM products ORDER BY name'
            )
            return [{'id': row[0], 'name': row[1], 'modified': row[2]} 
                    for row in cursor.fetchall()]
    
    def delete_product(self, product_id: str):
        """Delete product configuration"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM products WHERE product_id = ?', (product_id,))