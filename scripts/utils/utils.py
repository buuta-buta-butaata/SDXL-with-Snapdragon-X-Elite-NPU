import logging


logger = logging.getLogger(__name__)

def get_project_root():
    from pathlib import Path
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / ".here").exists():
            return parent
    raise FileNotFoundError("Project root not found.")

def print_imported():
    import sys
    logger.debug("Currently imported modules:")
    for name in sorted(sys.modules.keys()):
        logger.debug(name)

def check_torch_imported():
    import sys

    is_torch_imported = any(name == 'torch' or name.startswith('torch.') for name in sys.modules.keys())

    if not is_torch_imported:
        logger.debug("🎉 Perfect! 'torch' (PyTorch) has not been imported anywhere!")
    else:
        logger.debug("⚠️ Warning: 'torch' (PyTorch) has been imported somewhere.")
        for name in sorted(sys.modules.keys()):
            if 'torch' in name:
                logger.debug(f"  - {name}")

def _calc_cols_width(header, data, units):
    if header:
        items = zip(header, *data)
    else:
        items = zip(*data)

    cols_width = []
    for item in items:
        cols_width.append(max([len(str(x)) for x in item]))
    for i, unit in enumerate(units):
        if unit:
            cols_width[i] += len(unit) + 1
    return cols_width

def _get_align(align_id):
    align_dict = {
        "C": "^",
        "R": ">",
        "L": "<",
    }
    return align_dict[align_id.upper()]

def _generate_row_format(aligns, units):
    row_format = ""
    for i, align_id in enumerate(aligns):
        align = _get_align(align_id)
        if units:
            row_format += "| {:" + align + "{}} " + (units[i] if units[i] else "")
        else:
            row_format += "| {:" + align + "{}} "
    row_format += " |"
    return row_format
                
def print_table(title, subtitle, header, col_aligns, col_units, data):
    cols_width = _calc_cols_width(header, data, col_units)
    cols_num = len(cols_width)

    if not col_aligns:
        col_aligns = ["c"] * len(data[0])
    
    width = sum(cols_width) + 5 + 3 * (cols_num - 1)
    if title:
        logger.debug("=" * width)
        logger.debug(f" {title} ")
        logger.debug("=" * width)

    if subtitle:
        logger.debug(f" {subtitle} ")
        logger.debug("-" * width)

    row_format = _generate_row_format(col_aligns, None)
    if header:
        r = []
        for h, c in zip(header, cols_width):
            r.append([h, c])
        flattened_list = [item for sublist in r for item in sublist]
        logger.debug(row_format.format(*flattened_list))
        logger.debug("-" * width)

    row_format = _generate_row_format(col_aligns, col_units)
    if data:
        for col in data:
            r = []
            for i, (h, c) in enumerate(zip(col, cols_width)):
                unit_len = len(col_units[i]) if col_units[i] else 0
                r.append([h, c - unit_len])
            flattened_list = [item for sublist in r for item in sublist]
            logger.debug(row_format.format(*flattened_list))
            
    logger.debug("-" * width)
    
if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=logging.DEBUG
    )
    print_table("title", "sub", ["header_1", "header_2"], ["c", "r"], [None, "sec"], [[1, 2], [3, 4], [5, 6]])
    print_table("title", "sub", ["header_1", "header_2", "header_3"], ["l", "c", "r"], [None, "GB", "sec"],
                [[1, 2, 12], [3, 4, 34], [5, 6, 56]])
    print_table("title", None, ["header_1", "header_2", "header_3", "header_4"],
                ["l", "c", "r", "c"], [None, "GigaBytes", "sec", "%"], [[1, 2, 12, 123], [3, 4, 34, 345], [5, 6, 56, 456]])
    print_table("title", None, None,
                None, [None, None, None, None], [[1, 2, 12, 123], [3, 4, 34, 345], [5, 6, 56, 456]])
    
    
