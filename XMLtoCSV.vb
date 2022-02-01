Sub iterate()
    
    FolderName = "C:\Users\sarsaniv\Documents\"
    If Right(FolderName, 1) <> Application.PathSeparator Then FolderName = FolderName & Application.PathSeparator
    Fname = Dir(FolderName & "*.XML")

    'loop through the files
    Do While Len(Fname)


        Dim LstRw As Long
        Dim c As Integer
        Dim pFlPthSel
        Dim FlNmCSV As String
        Dim FndToC As Range, FndTrnCr As Range
        
    
        Workbooks.OpenXML Filename:=FolderName & Fname, Stylesheets:=Array(1)
        On Error GoTo 0
        ' Excel opens the XML file as a formatted CCD in the active worksheet
        With ActiveSheet
            ' Define the row number of the last populated cell
            LstRw = .Range("A65536").End(xlUp).Row
            ' Get rid of all hyperlinks
            .Cells.Hyperlinks.Delete
            ' Delete each blank row
            For c = LstRw To 1 Step -1
                With ActiveSheet.Range("A" & c)
                    If Len(.Value) = 0 And .End(xlToRight).Column > 255 Then
                        .EntireRow.Delete
                    End If
                End With
            Next c
            ' Find the table of contents row
            Set FndToC = .Range("a2:a" & LstRw).Find("Table of Contents", LookIn:=xlValues, LookAt:=xlWhole)
            If Not FndToC Is Nothing Then
                ' Find the last label in the table of contents ("Transfer of care")
                Set FndTrnCr = .Range("a2:a" & LstRw).Find("Transfer of care", LookIn:=xlValues, LookAt:=xlWhole)
                If Not FndTrnCr Is Nothing Then
                    ' Delete the entire table of contents rows
                    .Range(FndToC.Address & ":" & FndTrnCr.Address).EntireRow.Delete
                End If
            End If
        End With
        ' Define the name for the newly created file by replacing the "xml" extention with "csv"
        FlNmCSV = FolderName & Fname & ".csv"
        ' Save the active workbook as a csv
        ActiveWorkbook.SaveAs Filename:=FlNmCSV, FileFormat:=xlCSV, CreateBackup:=False
        ActiveWorkbook.Close SaveChanges:=True
        ' Close the active workbook
        ActiveWindow.Close
        Fname = Dir

    Loop
    
End Sub
