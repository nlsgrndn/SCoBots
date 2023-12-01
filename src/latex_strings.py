object_table = """\\begin{{subtable}}[b]{{\\textwidth}}\\centering
\\begin{{tabular}}{{@{{}}lccrr@{{}}}}
\\toprule
\\textbf{{Object/Entity}} & \\textbf{{Method}} & \\textbf{{Relevant}} & \\textbf{{Precision}} & \\textbf{{Recall}} \\\\
\\midrule
{0} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{{1}}}
\\end{{subtable}}
"""
figure = """
\\begin{{subfigure}}{{.53\\textwidth}}\\captionsetup{{aboveskip=-0.17em}}
  \\centering
  \\includegraphics[width=\\textwidth]{{{1}}}
  \\caption{{{0}}}
  \\label{{fig:{2}}}
\\end{{subfigure}}
"""
table_tex = """
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{{4}}}
\\hline
{0} \\\\
\\hline
{1} \\\\
\\hline
\\end{{tabular}}
\\caption{{{2}}}
\\label{{table:{3}}}
\\end{{table}}
"""
table_metric_tex = """
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{{4}}}
\\hline
\\multicolumn{{{6}}}{{|c|}}{{\\textbf{{{5}}}}} \\\\
\\hline
\\hline
{0} \\\\
\\hline
{1} \\\\
\\hline
\\end{{tabular}}
\\caption{{{2}}}
\\label{{table:{3}}}
\\end{{table}}
"""
qual_page = """\\newpage
\\thispagestyle{{empty}}
\\newgeometry{{left=1cm,bottom=1cm,right=1cm,top=1cm}}
    \\begin{{sidewaysfigure}}[htbp]
    \\centering
    \\centerline{{\\includegraphics[width=1\\textwidth]{{img_qual/0-separations_{0}.png}}}}
    \\caption{{{1}. The Columns from left to right: Input Image, Reconstruction, Foreground,
    Bounding Boxes, Background, K x Background Components, K x Background Masks, K x Background Color Maps, Alpha}}
    \\end{{sidewaysfigure}}
\\restoregeometry
"""