# Mentorship Graph
This project explores bachelor thesis abstracts and analyzes the relationships between students, their mentors, and commission members.
The system provides a predictive model that recommends the most suitable mentor for a given bachelor thesis abstract based on historical data and relationship patterns.

## Development
### Requirements
1. uv

### Project Setup
```bash
uv sync
```

### Data
1. Place the `committee.csv` file inside the `data` folder.
2. Split the dataset into train/validation/test sets: 
```bash
uv run python -m split_data
```
