import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import supabase

from db import _get_client # Import the function from db.py

# Attempt to get the Supabase client using the established function
supabase_client = _get_client()

# Check if the client was successfully initialized
if supabase_client is None:
    st.error("Failed to initialize Supabase client. Please check Supabase credentials.")
    st.stop() # Stop Streamlit execution if the client cannot be initialized

# No need to assign to st.session_state here if _get_client() handles it or
# if the client is used directly. If session state is desired for caching/persistence
# across reruns, it should be managed consistently, perhaps in app.py or db.py.
# For now, assume direct use of the returned client is sufficient.

def get_athletes():
    # Placeholder for fetching athletes. Should ideally come from db.py or Supabase.
    # For now, simulating a list of athletes.
    try:
        # Attempt to fetch from Supabase if a db module is available and has this function
        from db import cargar_atletas # Assuming this function exists
        athletes_df = cargar_atletas()
        return athletes_df['nombre'].tolist() if not athletes_df.empty else []
    except (ImportError, AttributeError):
        # Fallback to hardcoded list if db module or function is not available
        return ["Athlete A", "Athlete B", "Athlete C", "Athlete D"]

def fetch_sesiones_vmp(athlete_name: str = None, date_range: tuple = None):
    """Fetches sesiones_vmp data, with optional athlete and date filtering."""
    query = supabase_client.table("sesiones_vmp").select("*")
    
    if athlete_name:
        query = query.eq("atleta", athlete_name)
    
    if date_range and date_range[0] and date_range[1]:
        query = query.gte("fecha", date_range[0]).lte("fecha", date_range[1])
    
    try:
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            # Ensure date column is datetime type for proper filtering/display
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha']).dt.date
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching sesiones_vmp: {e}")
        return pd.DataFrame()

def fetch_wellness(athlete_name: str = None, date_range: tuple = None):
    """Fetches wellness data, with optional athlete and date filtering."""
    query = supabase_client.table("wellness").select("*")
    
    if athlete_name:
        query = query.eq("atleta", athlete_name)
    
    if date_range and date_range[0] and date_range[1]:
        query = query.gte("fecha", date_range[0]).lte("fecha", date_range[1])
    
    try:
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha']).dt.date
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching wellness: {e}")
        return pd.DataFrame()

def fetch_cargas_sesion(athlete_name: str = None, date_range: tuple = None):
    """Fetches cargas_sesion data, with optional athlete and date filtering."""
    query = supabase_client.table("cargas_sesion").select("*")
    
    if athlete_name:
        query = query.eq("atleta", athlete_name)
    
    if date_range and date_range[0] and date_range[1]:
        query = query.gte("fecha", date_range[0]).lte("fecha", date_range[1])
    
    try:
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha']).dt.date
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching cargas_sesion: {e}")
        return pd.DataFrame()

def update_supabase_row(table_name: str, row_id: str, data: dict):
    """Updates a single row in a Supabase table."""
    try:
        response = supabase_client.table(table_name).update(data).eq("id", row_id).execute()
        if response.data:
            st.success(f"Row {row_id} in {table_name} updated successfully.")
            return True
        else:
            st.error(f"Failed to update row {row_id} in {table_name}.")
            return False
    except Exception as e:
        st.error(f"Error updating row {row_id} in {table_name}: {e}")
        return False

def delete_supabase_row(table_name: str, row_id: str):
    """Deletes a single row from a Supabase table."""
    try:
        response = supabase_client.table(table_name).delete().eq("id", row_id).execute()
        if response.data:
            st.success(f"Row {row_id} from {table_name} deleted successfully.")
            return True
        else:
            st.error(f"Failed to delete row {row_id} from {table_name}.")
            return False
    except Exception as e:
        st.error(f"Error deleting row {row_id} from {table_name}: {e}")
        return False

def render_editor_tab(title, fetch_func, table_name):
    """Renders a Streamlit tab with filters and data editor."""
    st.subheader(f"{title} History & Editing")

    athletes = get_athletes()
    selected_athlete = st.selectbox(f"Select Athlete for {title}", ["All"] + athletes, key=f"{table_name}_athlete_select")

    # Date range picker
    today = datetime.now().date()
    one_year_ago = today - timedelta(days=365)
    
    date_range = st.date_input(
        f"Select Date Range for {title}",
        value=(one_year_ago, today),
        min_value=datetime(2000, 1, 1).date(),
        max_value=today,
        key=f"{table_name}_date_range",
        format="YYYY-MM-DD"
    )
    
    # Ensure date_range is a tuple of dates if valid
    valid_date_range = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        valid_date_range = (date_range[0], date_range[1])
    elif isinstance(date_range, datetime.date): # If only one date was selected and it's a date
        valid_date_range = (date_range, date_range)


    athlete_filter = selected_athlete if selected_athlete != "All" else None
    
    data_df = fetch_func(athlete_name=athlete_filter, date_range=valid_date_range)

    if not data_df.empty:
        # st.dataframe(data_df) # For debugging
        
        # Prepare DataFrame for st.data_editor
        # Ensure 'id' is preserved for updates/deletes, but not shown as editable if desired
        # Add a delete button column
        data_df['delete_action'] = False # Add a checkbox for deletion

        edited_df = st.data_editor(
            data_df,
            key=f"{table_name}_editor",
            use_container_width=True,
            num_rows="dynamic",
            # Hide the ID column if it exists and you don't want it editable, but keep it for updates
            column_order=["delete_action"] + [col for col in data_df.columns if col not in ["id", "delete_action"]] if 'id' in data_df.columns else data_df.columns.tolist(),
            # If you want to hide columns, you can do so here or in column_config
            column_config={
                "id": st.column_config.TextColumn("ID", disabled=True), # Hide ID
                "delete_action": st.column_config.CheckboxColumn("Delete?", help="Check to mark for deletion"),
                # Add other specific column configurations if needed (e.g., date formatting, numeric types)
                "fecha": st.column_config.DateColumn("Date", format="YYYY-MM-DD", disabled=False),
            }
        )

        # Handle updates and deletes after data editor interaction
        if st.button(f"Save Changes for {title}", key=f"save_{table_name}"):
            changes_made = False
            rows_to_delete = edited_df[edited_df['delete_action'] == True]
            rows_to_update = edited_df[edited_df['delete_action'] == False]

            # Process deletions first
            if not rows_to_delete.empty:
                for index, row in rows_to_delete.iterrows():
                    if 'id' in row and pd.notna(row['id']):
                        if delete_supabase_row(table_name, str(row['id'])):
                            changes_made = True
                    else:
                        st.warning("Cannot delete row without a valid ID.")

            # Process updates
            if not rows_to_update.empty:
                for index, row in rows_to_update.iterrows():
                    if 'id' in row and pd.notna(row['id']):
                        row_id = str(row['id'])
                        # Prepare data for update, excluding 'id' and 'delete_action'
                        update_data = row.drop(['id', 'delete_action'], errors='ignore').to_dict()
                        
                        # Convert pandas NaT to None for Supabase
                        for key, value in update_data.items():
                            if pd.isna(value) or value == '':
                                update_data[key] = None
                        
                        if update_supabase_row(table_name, row_id, update_data):
                            changes_made = True
                    else:
                        # Handle new rows if 'num_rows="dynamic"' allows creation
                        # This part requires more sophisticated handling:
                        # 1. Detect if a row has no 'id' but has data.
                        # 2. Call an `insert` function instead of `update`.
                        # For now, focusing on updates and deletes of existing rows.
                        st.warning("New row creation via data_editor is not explicitly handled in this example. Please use the appropriate insert functionality if needed.")

            if changes_made:
                st.rerun() # Rerun to refresh data after changes

    else:
        st.info(f"No data found for {title} with the current filters.")

def tab_historial():
    """Main function to render the History & Editing tab."""
    st.title("History & Editing")
    st.markdown("Manage and edit historical data from various modules.")

    # Create tabs for each data editor
    tab1, tab2, tab3 = st.tabs([
        "Sesiones VMP",
        "Wellness",
        "Cargas de Sesión"
    ])

    with tab1:
        render_editor_tab("Sesiones VMP", fetch_sesiones_vmp, "sesiones_vmp")

    with tab2:
        render_editor_tab("Wellness", fetch_wellness, "wellness")

    with tab3:
        render_editor_tab("Cargas de Sesión", fetch_cargas_sesion, "cargas_sesion")

# Example of how this tab would be called from app.py:
# if selected_tab == "Historial":
#     tab_historial()
