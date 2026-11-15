% vim: expandtab:sw=2:ts=2:
\NeedsTeXFormat{LaTeX2e}[2009/09/24]
\ProvidesClass{jcse}
          [2011/05/19 v0.1 Journal of Computing Science and Engineering (JCSE)]
\def\CJ@err#1{\ClassError{cj}{#1}\@eha}
\def\CJ@warn#1{\ClassWarningNoLine{cj}{#1}}
\DeclareOption{10pt}{}
\DeclareOption{11pt}{\CJ@warn{10pt used (not optional)}}
\DeclareOption{12pt}{\CJ@warn{10pt used (not optional)}}
\DeclareOption{twoside}{}
\DeclareOption{twocolumn}{}
\DeclareOption{onecolumn}{\CJ@warn{twocolumn used (not optional)}}
\DeclareOption{mathptm}{%
  \def\CJ@fonts{\RequirePackage{times,mathptm}}}
\DeclareOption{mathtime}{%
  \def\CJ@fonts{\RequirePackage{times,mathtime}}}
\DeclareOption{cmfonts}{%
  \let\CJ@fonts\relax}
\DeclareOption{epsfig}{\AtEndOfClass{\RequirePackage{epsfig}}}
\ExecuteOptions{cmfonts,epsfig}
\ProcessOptions
\LoadClass[a4paper,twoside,twocolumn]{article}
%\LoadClass[a4paper,twoside,twocolumn]{article}
% tolerances
\pretolerance=6000
\tolerance=9500
\hbadness=5000
\vbadness=5000
%\topmargin=0\p@
\emergencystretch=5pt
\sloppy
%
\RequirePackage{rotating}
\RequirePackage{color} % for color box
\RequirePackage{lettrine} % for PARstart command.
\RequirePackage{geometry} % Easier control over page layout.
\RequirePackage{afterpage} % Afterpage hook, for geometry of title page.

\def\@papertype{??? Paper}
\def\@papertypeAtFoot{??? Paper}
\def\@volumeyear{????}
\def\@DOI{????}
\def\@volumenumber{??}
\def\@issuenumber{??}
\def\@pp{???-???}

% page setup
\geometry{
  %voffset=1in, confuses geometry package.
  %hoffset=1in, confuses geometry package.
  %topskip=\baselineskip, nonexistent opt.
  % textheight=42\baselineskip, not needed anymore, using top/bototm instead.
  headsep=10mm,
  marginparsep=1mm,
  marginparwidth=5mm,
  headheight=26mm,
  footskip=12mm,
  columnsep=2pc,
  lmargin=20mm,
  rmargin=20mm,
  top=25mm,
  bottom=25mm
}

\CJ@fonts

% new permission statement. Text only!
\long\def\@permission{
Copyright\copyright \@volumeyear\xspace by The Korean Institute of Information Scientists and Engineers\,(KIISE).
Permission to make digital or hard copies of part or all of this work for personal or classroom use
is granted without fee provided that copies are not made or distributed for profit or commercial
advantage and that copies bear this notice and the full citation on the first page. Permission to
post author-prepared versions of the work on author's personal web pages or on the noncommercial
servers of their employer is granted without fee provided that the KIISE citation and notice of the
copyright are included. Copyrights for components of this work owned by authors other than
KIISE must be honored. Abstracting with credit is permitted. To copy otherwise, to republish, to
post on servers, or to redistribute to lists, requires an explicit prior permission and/or a fee.
Request permission to republish from:\,JCSE Editorial Office, KIISE. FAX +82 2 521 1352 or email
{\tt office@kiise.org}. The Office must receive a signed hard copy of the Copyright form.
}

\def\permission#1{%
  \gdef\@permission{#1}
}

\def\@permissionUrl{http://jcse.kiise.org}
\def\permissionUrl#1{
  \gdef\@permissionUrl{#1}
}

\def\@copyrightString{copyright xxx}
\def\copyrightString#1{
  \gdef\@copyrightString{#1}
}

\def\@issn{pISSN: xxx eISSN: yyy}
\def\issn#1{
  \gdef\@issn{#1}
}

\newcommand{\thline}{\noalign{\hrule height 0.08em}}
\newfont{\papertyfnt}{pzcmi at 20pt}

\def\shortauthors#1{\gdef\@shortauthors{#1}}
\def\title{\@dblarg\CJ@title}
\def\CJ@title[#1]#2{%
  \def\@title{#2}%
  \def\@shorttitle{#1}}
\def\@title{\CJ@warn{No title}}
\def\@shorttitle{}
%\newcounter{affil}
%\newcounter{remaffil}
%\def\author#1{%
%  \g@addto@macro\@authors{%
%    \@andorcomma,{\ignorespaces#1\unskip\affil@mark}}%
%  \def\CJ@noaffil{\CJ@warn{No affiliation specified for #1}}}
%\def\CJ@noaffil{%
% \CJ@err{No author specified}%
% \global\let\@andorcomma\relax
%  \@gobbletwo}
%\def\@authors{\relax\@gobbletwo}
%\def\and{\CJ@err{\protect\and\space is not supported}}
%\def\@andorcomma#1#2\@andorcomma#3{%
%  \ifx#3\relax
%    \ifx#1\relax%
%      \let\@tempa\relax
%    \else
%    \def\@tempa{\space\andname\space #2\@gobbletwo}%
%    \fi
%  \else
%    \def\@tempa{#1 #2\@andorcomma#3}%
%  \fi
%  \@tempa}
%\def\affiliation#1{%
%  \def\@tempa{#1}%
%  \@tempswatrue
%  \@affils
%  \if@tempswa
%    \stepcounter{affil}\setcounter{remaffil}{\theaffil}%
%    \edef\@tempa##1{%
%      \noexpand\g@addto@macro\noexpand\@affils{%
%        \noexpand\doaffil{\theaffil}{##1}}}%
%    \@tempa{#1}%
%    \edef\@tempa{\noexpand\CJ@addaffil{\theaffil}}%
%  \fi
%  \expandafter\@tempa\@authors\relax\relax
%  \let\CJ@noaffil\relax}
%\let\@affils\@empty
%\def\CJ@addaffil#1#2\relax#3\relax{%
%  \def\@authors{#2\def\theaffil{#1}#3\relax}}
%\def\doaffil#1#2{%
%  \def\@tempb{#2}%
%  \ifx\@tempa\@tempb
%    \@tempswafalse
%    \def\@tempa{\CJ@addaffil{#1}}%
%  \fi}
%\def\address#1{%
%  \g@addto@macro\@authors{\textsuperscript{$*$}}%
%  \def\@address{#1}%
%  \def\address##1{%
%    \CJ@err{You may only specify one correspondence address}}}
%\def\@address{\CJ@warn{No correspondence address specified}}
%
%
%%SDH David's \email puts email addresses at the bottom of the page--wrong!
%\def\email#1{%
%  \g@addto@macro\@address{\endgraf\noindent(e-mail: #1)}%
%  \def\email##1{%
%    \CJ@err{You may only specify one correspondence email}}}
%\def\email#1{%
%\def\@email{Email: #1}%
%}
%\def\authname#1{{\small\bfseries #1}}
\newfont{\authname}{ptmb8t at 9pt}
\newfont{\affaddr}{ptmr at 9pt}
\newfont{\emailfont}{ptmb8t at 9pt}
\def\email#1{~~~~\emailfont{#1}}
%
\let\affil@mark\@empty
\def\keywords#1{%
  \gdef\@keywords{%
    \addvspace\medskipamount
    \noindent\strut{\sffamily\bfseries Keywords:\/} \ignorespaces#1\unskip}
}
\def\@keywords{\ClassWarningNoLine{cj}{No keywords}}
\def\category#1{%
  \gdef\@category{%
    \addvspace\medskipamount
    \noindent\strut{\sffamily\bfseries Category:\/} \ignorespaces#1\unskip}
}
\def\categories#1{%
  \gdef\@category{%
    \addvspace\medskipamount
    \noindent\strut{Categories:\/} \ignorespaces#1\unskip}
}
\def\@category{\ClassWarningNoLine{cj}{No category specified}}

\def\received#1{%
  \gdef\@received{Received #1}}
\def\@received{\ClassWarningNoLine{cj}{No received date}}
\def\revised#1{%
  \gdef\@revised{, Revised #1}} %SDH
\let\@revised\@empty
\def\accepted#1{%
  \gdef\@accepted{Accepted #1}}
\def\@accepted{\CJ@warn{No accepted date}}
\newbox\absbox
\def\abstract{\lrbox\absbox\minipage{\textwidth}%
  %\hrule\@height.2pt
  \vspace{5mm}
  {\sffamily\bfseries Abstract}
  \par
  \normalfont\small
  \parfillskip\stretch2}
\def\endabstract{\endminipage\endlrbox}

\def\titlepagelayout{
  % Layout for title page
  \newgeometry{
    %voffset=1in, confuses geometry package.
    %hoffset=1in, confuses geometry package.
    %topskip=\baselineskip, nonexistent opt.
    % textheight=42\baselineskip, not needed anymore, using top/bototm instead.
    headsep=4mm,
    marginparsep=1mm,
    marginparwidth=5mm,
    headheight=26mm,
    footskip=10mm,
    columnsep=2pc,
    lmargin=20mm,
    rmargin=20mm,
    top=40mm,
    bottom=60mm
  }
}

\def\restorelayout {
  % Because one column is a PAGE in LaTeX system,
  % we restore page after TWO pages!
  \afterpage{
    \afterpage {
      \aftergroup
      \restoregeometry
    }
  }
}

\def\@bottomstuff{}
\def\bottomstuff#1{
  \gdef\@bottomstuff{#1}
}

%
% maketitle command
%
\def\maketitle{
  \titlepagelayout
{%
  \long\def\@makefntext##1{\parindent 1em\noindent
    \@textsuperscript{\normalfont\@thefnmark}##1}%
  \twocolumn[%
    \thispagestyle{title}\label{FirstPage}%

    \normalfont %\centering
    {
      \vspace{1cm}
      \parbox[b]{\textwidth}{ %\centering
        {\sffamily\bfseries\LARGE\@title\par}%
        \addvspace\bigskipamount
        \@author
        \usebox\absbox\par
       \addvspace\bigskipamount
       \small %\itshape
       \@category
     }\par
     \parbox{\textwidth}{ %\centering
       \@keywords\par
       \addvspace\bigskipamount
    }
  }%
  %\hrule\@height.5pt
  \vspace{1pt}%
  ]%
  %{\def\thefootnote{\@fnsymbol\c@footnote}%
  % \footnotetext[1]{\def\\{\unskip, \ignorespaces}\@address}%
  % \@thanks}%
  \suppressfloats[t]}%
  \let\and\relax
  \let\affiliation\relax
  \let\author\relax
  \let\@affils\relax
  \let\keywords\relax
  \let\@keywords\relax
  \let\address\relax
  \let\email\relax
  \let\@address\relax
  \let\maketitle\relax
  \let\thanks\@gobble

  \restorelayout
}

\def\ps@cj{%

  \def\@oddhead{%
    \CJ@ruledhead
      {}
      {}
      {\sffamily\footnotesize \@shorttitle}
  }%
  \def\@evenhead{%
    \CJ@ruledhead
      {\sffamily\footnotesize\itshape \journalname, \volumename \@volumenumber,
        \numbername \@issuenumber, \@volumeyear. pp. \@pp
      }
      {}
      {}
  }%

%  \def\@evenhead{%
%    \vbox to \headheight{\vss
%   \let\textsuperscript\@gobble
%    \CJ@ruledhead
%      \thepage
%%     {\@authors\@andorcomma\relax\@andorcomma\relax}%
%     {\@shortauthors : {\@shorttitle}}%
%      \relax}}%

  \def\@oddfoot{%
    \CJ@ruledhead
      {\sffamily\small \@shortauthors}
      {\sffamily \thepage}
      {\sffamily\small \@permissionUrl}
  }%
  \def\@evenfoot {
    \CJ@ruledhead
      {\sffamily\scriptsize DOI: \@DOI}
      {\sffamily \thepage}
      {\sffamily\scriptsize \@shortauthors}
  }
}
%       \textsc\journalname,\quad\volumename\space\@volumenumber,\quad
%         \numbername\space\@issuenumber,\quad\@volumeyear}\relax}%

% Page style of the first page (=title page)
\def\ps@title{%
  \def\@oddhead{%
    \begin{tabular*}{\textwidth}{p{7cm}p{3cm}r}
      \epsfig{figure=logo.eps}
      &
      &
      \parbox{6cm}{
        \begin{flushright}
        \vspace{-2cm}
        \footnotesize\sffamily\bfseries\itshape
        \colorbox{black}{\papertyfnt \normalsize \color{white}{~\@papertype}~} \\
        \vspace{1mm}
        \journalname \\
        Vol.\@volumenumber, No.\@issuenumber, \@volumeyear, pp. \@pp
        \end{flushright}
      }
    \end{tabular*}
    %\vbox{%
    %  \hbox to 0.5\textwidth{%
    %    \epsfig{figure=logo.eps}}
    %  \hbox to 0.5\textwidth{%
    %    \journalname,\\Vol \end{minipage}}
    %}
  }%

  \let\@evenhead\@oddhead

  \def\@oddfoot{%
    \vbox{
      \vspace{5mm}
      \hrule\@height.5pt
      \vspace{5mm}
      \hbox to \textwidth {
        \parbox{\textwidth}{
          \colorbox{black}{
            \sffamily\bfseries
            \color{white}{\@papertypeAtFoot}
          }
          { \small\sffamily\bfseries \@DOI }
          { \hfill \small\sffamily\bfseries \@permissionUrl }

          \vspace{1mm}
          {\footnotesize \@permission }

          {
            \addvspace\medskipamount
            \footnotesize
            \@received, \@accepted \@revised\\
            \@bottomstuff
          }

          \vspace{5mm}
          { \footnotesize\sffamily \@copyrightString }
          { \hfill \footnotesize\sffamily \@issn }
        }
      }
    }

  %\CJ@ruledhead\relax{% foot for first page
  %  \textsc\journalname, \quad\volumename\space\@volumenumber,
  %  \quad\numbername\space\@issuenumber, \quad\@volumeyear}\relax
  }%

  \let\@evenfoot\@oddfoot
}

\ps@cj

\def\CJ@ruledhead#1#2#3{%
  \vbox{%
    %\hrule\@height.5pt
    \hbox to\textwidth
    {
      \normalfont\vrule\@width\z@\@height 9.5pt\@depth 4.5pt%
      \rlap{#1}%
      {\hfil#2\hfil}%
      \llap{#3}
    }
    %\hrule\@height.5pt
  }
}
\ps@cj
\def\@startpage{\pageref{FirstPage}}
\def\@endpage{\pageref{LastPage}}
\def\papertype#1{\gdef\@papertype{#1}}
\def\papertypeAtFoot#1{\gdef\@papertypeAtFoot{#1}}
\def\volumeyear#1{\gdef\@volumeyear{#1}}
\def\DOI#1{\gdef\@DOI{#1}}
\def\volumenumber#1{\gdef\@volumenumber{#1}}
\def\pp#1{\gdef\@pp{#1}}
\def\issuenumber#1{\gdef\@issuenumber{#1}}
\def\startpage#1{\gdef\@startpage{#1}\c@page#1}
\def\endpage#1{\gdef\@endpage{#1}}
\def\lastpage@putlabel{%
  \advance\c@page\m@ne
   \immediate\write\@auxout{\string
   \newlabel{LastPage}{{}{\thepage}}}%
   \advance\c@page\@ne}
\AtEndDocument{%
   \clearpage\lastpage@putlabel}%
\def\enumerate{\ifnum \@enumdepth >3 \@toodeep\else
    \advance\@enumdepth \@ne 
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}\list
    {\csname label\@enumctr\endcsname}{\usecounter
    {\@enumctr}\def\makelabel##1{##1\hss}%
    \labelsep=3\p@\parsep=0pt\listparindent=0pt\itemsep=0pt\partopsep=0pt
    \rightmargin=0pt}\fi}
%
\def\itemize{\ifnum \@itemdepth >3 \@toodeep\else \advance\@itemdepth \@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \list{\csname\@itemitem\endcsname}{\def\makelabel##1{##1\hss}%
    \labelsep=3\p@\parsep=0pt\listparindent=0pt\itemsep=0pt\partopsep=0pt
    \rightmargin=0pt}\fi}
%
\setlength\parskip{0pt plus 0.1pt}
\setlength\parindent{1em}
\setlength\topsep{8pt plus 2pt minus 4pt}
\setlength\partopsep{0pt plus 2pt}
\setlength\itemsep{0pt plus 1pt}


%%%
%%% Section numbering and font configuration
%%%
\def\@seccntformat#1{
  \csname da#1\endcsname
  \csname adddot@#1\endcsname\ 
}
\def\section{%
  \@startsection
    {section}{1}{\z@}{-14\p@ \@plus -2\p@ \@minus -2\p@}%
    {6\p@ plus 2\p@}{\sffamily\normalsize\bfseries\MakeUppercase}}
\def\adddot@section{.}
\def\enddot@section{}
\def\subsection{%
  \@startsection
    {subsection}{2}{\z@}{-14\p@ \@plus -2\p@ \@minus -2\p@}%
    {6\p@ plus 2\p@}{\sffamily\normalsize\itshape}}
\def\adddot@subsection{.}
\def\enddot@subsection{}
\def\subsubsection{%
  \@startsection
    {subsubsection}{3} % name, level
    {4mm} % indent
    {12\p@} % beforeskip
    {1\p@ plus 0\p@} % afterskip
    {\normalfont\normalsize\itshape} % style
  }
\def\adddot@subsubsection{)}
\def\enddot@subsubsection{:}
\def\paragraph{%
  \@startsection
    {paragraph}{4}{\z@}{12\p@}%
    {-1em}{\normalfont\normalsize\itshape}}
\def\subparagraph{%
  \@startsection
    {paragraph}{5}{\z@}{12\p@}%
    {-1em}{\normalfont\normalsize\itshape}}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\def\fps@eqnfloat{!tp}
\def\ftype@eqnfloat{4}
\newenvironment{eqnfloat}
               {\@float{eqnfloat}}
               {\end@float}
\newenvironment{eqnfloat*}
               {\@dblfloat{eqnfloat}}
               {\end@dblfloat}
\renewenvironment{table}
               {\@float{table}\small\ignorespaces}
               {\end@float}
\renewenvironment{table*}
               {\@dblfloat{table}\small\ignorespaces}
               {\end@dblfloat}

% Caption Font
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\bfseries\large#1.} \sffamily\normalsize#2}%
  \ifdim \wd\@tempboxa >\hsize
    {\bfseries\small#1.} \small#2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil} % remove hfil for l/r justfication.
  \fi
  \vskip\belowcaptionskip}

% Howto print "figure" and "table".
\def\figurename{Fig.}
\def\fnum@figure{\figurename~\thefigure}
\def\fnum@table{\tablename~\thetable}
\def\thetable{\Roman{table}}




\RequirePackage{amsthm}
\newtheoremstyle{theor}
    {6pt}% above space (default)
    {6pt}% below space
    {\itshape}% body
    {1em}% indent
    {\scshape}% head
    {.}% punct
    { }% space
    {}% custom
\theoremstyle{theor}

% Section numbering... used by \ref command and others.
\def\thesection{\Roman{section}}
\def\thesubsection{\thesection-\mbox{\Alph{subsection}}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection\alph{paragraph}}
% used by seccntformat command.
\def\dasection{\Roman{section}}
\def\dasubsection{\mbox{\Alph{subsection}}}
\def\dasubsubsection{\arabic{subsubsection}}
\def\daparagraph{\thesubsubsection\alph{paragraph}}

\newtheorem{definition}{Definition}
\def\thedefinition{\arabic{definition}}
\newtheorem{lemma}{Lemma}
\def\thelemma{\arabic{lemma}}
\newtheorem{proposition}{Proposition}
\def\theproposition{\arabic{proposition}}
\newtheorem{corollary}{Corollary}
\def\thecorollary{\arabic{corollary}}
\newtheorem{axiom}{Axiom}
\def\theaxiom{\arabic{axiom}}

\newtheoremstyle{remark}
    {6pt}% above space (default)
    {6pt}% below space
    {\normalfont}% body
    {1em}% indent
    {\scshape}% head
    {.}% punct
    { }% space
    {}% custom
\theoremstyle{remark}

\newtheorem{remark}{Remark}
\newtheorem{example}{Example}

{\theoremstyle{theor}
\newtheorem{theorem}{Theorem}
\def\thetheorem{\arabic{theorem}}
}

\providecommand{\textsuperscript}[1]{%
  \ensuremath{{}^{\mbox{\normalfont\scriptsize#1}}}}
\providecommand{\@textsuperscript}[1]{%
  \ensuremath{{}^{\mbox{\normalfont\scriptsize#1}}}}
\providecommand\MakeUppercase[1]{%
  \protected@edef\@tempa{\uppercase{#1}}\@tempa}
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #6{%
        \@hangfrom{\hskip #3\relax\@svsec}%
          \interlinepenalty \@M #8\csname enddot@#1\endcsname\@@par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #7}%
  \else
    \def\@svsechd{%
      #6{\hskip #3\relax
      \@svsec #8\csname enddot@#1\endcsname}%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}}%
  \fi
  \@xsect{#5}}
\def\@ssect#1#2#3#4#5{%
  \@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #4{%
        \@hangfrom{\hskip #1}%
          \interlinepenalty \@M #5\@@par}%
    \endgroup
  \else
    \def\@svsechd{#4{\hskip #1\relax #5}}%
  \fi
  \@xsect{#3}}
\def\journalname{Journal of Computing Science and Engineering}
\def\andname{and}
\def\ppname{pp}
\def\numbername{No.\@}
\def\volumename{Vol.\@}

%extra macros

% Inserts space after macro unless followed by punctuation.
\def\@xspace{%
  \ifx\@let@token\bgroup\else%
  \ifx\@let@token\egroup\else%
  \ifx\@let@token\/\else%
  \ifx\@let@token\ \else%
  \ifx\@let@token~\else%
  \ifx\@let@token.\else%
  \ifx\@let@token!\else%
  \ifx\@let@token,\else%
  \ifx\@let@token:\else%
  \ifx\@let@token;\else%
  \ifx\@let@token?\else%
  \ifx\@let@token'\else%
  \ifx\@let@token)\else%
  \ifx\@let@token-\else%
  \ifx\@let@token$\else%
  \ifx\@let@token/\else%
   \space%
   \fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi}
\newcommand{\xspace}{\futurelet\@let@token\@xspace}

% Omit extra full stop after et al. if at end of sentence
\def\@etstop{%
  \ifx\@let@token.%
   \else.%
  \fi\xspace}
\def\etstop{\futurelet\@let@token\@etstop}
\def\etal{{\it et al}\etstop}

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\DeclareMathAlphabet\mathbit
    \encodingdefault\rmdefault\bfdefault\itdefault
\DeclareOldFontCommand{\bi}{\normalfont\bfseries\itshape}{\mathbit}

%Appendix commands
\def\appendix{\@@par
 \@addtoreset{table}{section}        
 \@addtoreset{figure}{section} 
 \@addtoreset{equation}{section} 
 \setcounter{section}{0}
 \setcounter{subsection}{0}
 \setcounter{subsubsection}{0}
 \setcounter{equation}{0}
 \setcounter{figure}{0}
 \setcounter{table}{0}
 \def\thesection{Appendix~\Alph{section}}   
\def\thelemma{\Alph{section}.\arabic{lemma}}
\def\thedefinition{\Alph{section}.\arabic{definition}}
\def\thetheorem{\Alph{section}.\arabic{theorem}}
\def\theproposition{\Alph{section}.\arabic{proposition}}
\def\thecorollary{\Alph{section}.\arabic{corollary}}
\def\theaxiom{\Alph{section}.\arabic{axiom}}
 \def\theequation{%
      \Alph{section}.\arabic{equation}}  
 \def\thetable{%              
      \Alph{section}.\Roman{table}}
 \def\thefigure{%
      \Alph{section}.\arabic{figure}}
}

\newcommand{\centre}[2]{\multispan{#1}{\hfill #2\hfill}}
\newcommand{\crule}[1]{\multispan{#1}{\hrulefill}}

\def\lo#1{\llap{${}#1{}$}}

\def\tabnote#1{\par\rlap{\begin{minipage}[l]{\hsize}
\small#1
\end{minipage}}\\}

\def\lineup{\def\0{\hbox{\phantom{\small\rm 0}}}%
    \def\m{\hbox{$\phantom{-}$}}%
    \def\-{\llap{$-$}}}

\newcommand{\ms}{\noalign{\vspace{3\p@ plus2\p@ minus1\p@}}}
\newcommand{\ns}{\noalign{\vspace{-3\p@ plus-1\p@ minus-1\p@}}}

%%%%%%%%%%%%%%%%%%%%%
% Tables rules      %
%%%%%%%%%%%%%%%%%%%%%

\newcommand{\boldarrayrulewidth}{1\p@} 
% Width of bold rule in tabular environment.

\def\bhline{\noalign{\ifnum0=`}\fi\hrule \@height  
\boldarrayrulewidth \futurelet \@tempa\@xhline}

\def\@xhline{\ifx\@tempa\hline\vskip \doublerulesep\fi
      \ifnum0=`{\fi}}

%
% Rules for tables with extra space around
%
\newcommand{\br}{\ms\bhline\ms}
\newcommand{\mr}{\ms\hline\ms}

\newcommand{\case}[2]{{\textstyle\frac{#1}{#2}}}

\newcommand{\ack}{\section*{Acknowledgements}}
\newcommand{\ackn}{\section*{Acknowledgement}}

% need to be explicit with `References' for \uppercase'ing
\def\thebibliography#1{\section*{References}\small\list
 {[\arabic{enumi}]}{\labelsep 0.5em%
\hyphenpenalty=50\itemsep=-1pt
 \settowidth\labelwidth{[#1]}\leftmargin\labelwidth
 \advance\leftmargin\labelsep
 \usecounter{enumi}}
 \def\newblock{\hskip .14em plus .38em minus -.1em}
 \sloppy
 \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

\def\@cite#1#2{[{#1\if@tempswa , #2\fi}]}
\def\@biblabel#1{[#1]}

\let\origthebibliography\thebibliography
\def\thebibliography#1{\origthebibliography{\hbox to 1em{\hss}}}

\hyphenpenalty=20
\renewcommand{\floatpagefraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.95}
\renewcommand{\topfraction}{0.95}
\renewcommand{\dbltopfraction}{0.95}
%\renewcommand{\topnumber}{3}
%\renewcommand{\dbltopnumber}{3}


% paragraph starting
\def\PARstart#1#2{ \lettrine{\bf #1}{#2} }

\endinput
%% End of file `jcse.cls'.
