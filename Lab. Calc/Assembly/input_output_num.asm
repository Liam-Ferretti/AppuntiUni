;SOMMA TRA DUE NUMERI INTERI IN ASM

section .data
    msg1    db "Inserisci il primo numero (max 10 cifre): ", 0
    len1    equ $ - msg1
    msg2    db "Inserisci il secondo numero (max 10 cifre): ", 0
    len2    equ $ - msg2
    msgRes  db "La somma e': ", 0
    lenRes  equ $ - msgRes
    newline db 0Ah 

section .bss 
    buffer resb 10 
    n1 resd 1
    n2 resd 1 

section .text 
    global _start

_start:
    ; --- Stampa prompt 1 ---
    mov eax, 4
    mov ebx, 1
    mov ecx, msg1
    mov edx, len1
    int 0x80

    ; --- Leggi primo numero ---
    mov eax, 3
    mov ebx, 0
    mov ecx, n1
    mov edx, 10 
    int 80h

    ; --- Converti n1 ---
    xor eax, eax
    xor ebx, ebx
    mov esi, n1
.loop1:
    mov bl, [esi]
    cmp bl, 0x0A
    je .done1
    sub bl, '0'
    imul eax, eax, 10
    add eax, ebx
    inc esi
    jmp .loop1
.done1:
    mov [n1], eax

    ; --- Stampa prompt 2 ---
    mov eax, 4
    mov ebx, 1
    mov ecx, msg2
    mov edx, len2
    int 0x80

    ; --- Leggi secondo numero ---
    mov eax, 3
    mov ebx, 0
    mov ecx, n2
    mov edx, 10
    int 80h

    ; --- Converti n2 ---
    xor eax, eax
    xor ebx, ebx
    mov esi, n2
.loop2:
    mov bl, [esi]
    cmp bl, 0x0A
    je .done2
    sub bl, '0'
    imul eax, eax, 10
    add eax, ebx
    inc esi
    jmp .loop2
.done2:

    ; --- Somma ---
    add  eax, [n1]

    ; --- Converti numero in ASCII ---
    mov edi, buffer + 10
    mov ecx, 10
.conv_loop:
    dec edi
    xor edx, edx
    div ecx
    add dl, '0'
    mov [edi], dl
    test eax, eax
    jnz .conv_loop

    ; --- Stampa ---

    mov eax, 4
    mov ebx, 1
    mov ecx, msgRes
    mov edx, lenRes
    int 80h

    mov eax, 4
    mov ebx, 1
    mov ecx, edi
    mov edx, buffer + 10
    sub edx, edi
    int 80h

    ; --- Newline ---
    mov eax, 4
    mov ebx, 1
    mov ecx, newline
    mov edx, 1
    int 80h

    ; --- Esci ---
    mov eax, 1
    xor ebx, ebx
    int 80h