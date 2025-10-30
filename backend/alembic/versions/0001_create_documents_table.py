"""create documents table

Revision ID: 0001
Revises: 
Create Date: 2025-10-14 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('doc_metadata', postgresql.JSONB, nullable=True),
        sa.Column('embedding', Vector(768))
    )

def downgrade():
    op.drop_table('documents')
