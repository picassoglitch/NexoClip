"""Integrations with external systems (Nexo AI is the first).

Each integration lives in its own subpackage and exports a clean service
surface so route handlers stay thin. See nexo_ai/ for the operator-dashboard
integration that provisions tenants and validates SSO redirects.
"""
